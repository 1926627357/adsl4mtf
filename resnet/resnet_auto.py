# coding=utf-8
# Copyright 2020 The Mesh TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MNIST using Mesh TensorFlow and TF Estimator.
This is an illustration, not a good model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
from dataset import load_dataset  # local file import
import tensorflow.compat.v1 as tf
import mesh_tensorflow.auto_mtf
import os
from model import resnet_model
from config import *
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.flags.DEFINE_string("data_dir", "/home/haiqwa/dataset/cifar10",
                       "Path to directory containing the MNIST dataset")
tf.flags.DEFINE_string("model_dir", "/home/haiqwa/output/resnet", "Estimator model_dir")
tf.flags.DEFINE_integer("batch_size", 64,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("hidden_size", 512, "Size of each hidden layer.")
tf.flags.DEFINE_integer("train_epochs", 2, "Total number of training epochs.")
tf.flags.DEFINE_integer("epochs_between_evals", 1,
                        "# of epochs between evaluations.")
tf.flags.DEFINE_integer("eval_steps", 0,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
tf.flags.DEFINE_string("mesh_shape", "b1:2;b2:2", "mesh shape")
tf.flags.DEFINE_string("layout", "row_blocks:b1;col_blocks:b2",
                       "layout rules")

FLAGS = tf.flags.FLAGS


def mnist_model(image, labels, mesh):
	"""The model.
	Args:
		image: tf.Tensor with shape [batch, 28*28]
		labels: a tf.Tensor with shape [batch] and dtype tf.int32
		mesh: a mtf.Mesh
	Returns:
		logits: a mtf.Tensor with shape [batch, 10]
		loss: a mtf.Tensor with shape []
	"""
	batch_dim = mtf.Dimension("batch", FLAGS.batch_size)
	rows_dim = mtf.Dimension("rows_size", image_height)
	cols_dim = mtf.Dimension("cols_size", image_width)
	channel_dim = mtf.Dimension("image_channel", num_channels)
	classes_dim = mtf.Dimension(name='classesnum',size=classesnum)
	x = mtf.import_tf_tensor(
		mesh, tf.reshape(image, [FLAGS.batch_size, image_height, image_width, num_channels]),
		mtf.Shape(
			[batch_dim, rows_dim, cols_dim, channel_dim]))
	x = mtf.transpose(x, [batch_dim, rows_dim, cols_dim, channel_dim])
	# print(x.shape)
	logits = resnet_model(x, classes_dim=classes_dim,depth=50)


	if labels is None:
		loss = None
	else:
		labels = mtf.import_tf_tensor(
			mesh, tf.reshape(labels, [FLAGS.batch_size]), mtf.Shape([batch_dim]))
		loss = mtf.layers.softmax_cross_entropy_with_logits(
			logits, mtf.one_hot(labels, classes_dim), classes_dim)
		loss = mtf.reduce_mean(loss)
	return logits, loss


def model_fn(features, labels, mode, params):
	"""The model_fn argument for creating an Estimator."""
	tf.logging.info("features = %s labels = %s mode = %s params=%s" %
					(features, labels, mode, params))
	global_step = tf.train.get_global_step()
	graph = mtf.Graph()
	mesh = mtf.Mesh(graph, "my_mesh")
	logits, loss = mnist_model(features, labels, mesh)
	
	variables = graph._all_variables
	tf.logging.info("[variable num]: {}".format(len(variables)))
	for v in variables:
		tf.logging.info("[variable] (name, shape): ({},{})".format(v.name,v.shape))

	mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
	layout_rules = mtf.auto_mtf.layout(graph, mesh_shape, [logits, loss])
	tf.logging.info("[auto mtf search] strategy: {}".format(layout_rules))
	
	mesh_size = mesh_shape.size
	mesh_devices = ["gpu:0", "gpu:1", "gpu:2", "gpu:3"]
	mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
		mesh_shape, layout_rules, mesh_devices)

	if mode == tf.estimator.ModeKeys.TRAIN:
		var_grads = mtf.gradients(
			[loss], [v.outputs[0] for v in graph.trainable_variables])
		optimizer = mtf.optimize.AdafactorOptimizer()
		update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)

	lowering = mtf.Lowering(graph, {mesh: mesh_impl})
	restore_hook = mtf.MtfRestoreHook(lowering)

	tf_logits = lowering.export_to_tf_tensor(logits)
	if mode != tf.estimator.ModeKeys.PREDICT:
		tf_loss = lowering.export_to_tf_tensor(loss)
		tf.summary.scalar("loss", tf_loss)

	if mode == tf.estimator.ModeKeys.TRAIN:
		tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
		tf_update_ops.append(tf.assign_add(global_step, 1))
		train_op = tf.group(tf_update_ops)
		saver = tf.train.Saver(
			tf.global_variables(),
			sharded=True,
			max_to_keep=10,
			keep_checkpoint_every_n_hours=2,
			defer_build=False, save_relative_paths=True)
		tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
		saver_listener = mtf.MtfCheckpointSaverListener(lowering)
		saver_hook = tf.train.CheckpointSaverHook(
			FLAGS.model_dir,
			save_steps=1000,
			saver=saver,
			listeners=[saver_listener])

		accuracy = tf.metrics.accuracy(
			labels=labels, predictions=tf.argmax(tf_logits, axis=1))

		# Name tensors to be logged with LoggingTensorHook.
		tf.identity(tf_loss, "cross_entropy")
		tf.identity(accuracy[1], name="train_accuracy")

		# Save accuracy scalar to Tensorboard output.
		tf.summary.scalar("train_accuracy", accuracy[1])

		# restore_hook must come before saver_hook
		return tf.estimator.EstimatorSpec(
			tf.estimator.ModeKeys.TRAIN, loss=tf_loss, train_op=train_op,
			training_chief_hooks=[restore_hook, saver_hook])

	if mode == tf.estimator.ModeKeys.PREDICT:
		predictions = {
			"classes": tf.argmax(tf_logits, axis=1),
			"probabilities": tf.nn.softmax(tf_logits),
		}
		return tf.estimator.EstimatorSpec(
			mode=tf.estimator.ModeKeys.PREDICT,
			predictions=predictions,
			prediction_hooks=[restore_hook],
			export_outputs={
				"classify": tf.estimator.export.PredictOutput(predictions)
			})
	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(
			mode=tf.estimator.ModeKeys.EVAL,
			loss=tf_loss,
			evaluation_hooks=[restore_hook],
			eval_metric_ops={
				"accuracy":
				tf.metrics.accuracy(
					labels=labels, predictions=tf.argmax(tf_logits, axis=1)),
			})


def run_mnist():
	"""Run MNIST training and eval loop."""
	mnist_classifier = tf.estimator.Estimator(
		model_fn=model_fn,
		model_dir=FLAGS.model_dir)

	# Set up training and evaluation input functions.
	def train_input_fn():
		"""Prepare data for training."""

		# When choosing shuffle buffer sizes, larger sizes result in better
		# randomness, while smaller sizes use less memory. MNIST is a small
		# enough dataset that we can easily shuffle the full epoch.
		ds = load_dataset(FLAGS.data_dir)
		# ds = dataset.train(FLAGS.data_dir)
		ds_batched = ds.cache().shuffle(buffer_size=50000).batch(FLAGS.batch_size,drop_remainder=True)

		# Iterate through the dataset a set number (`epochs_between_evals`) of times
		# during each training session.
		ds = ds_batched.repeat(FLAGS.epochs_between_evals)
		return ds

	# def eval_input_fn():
	# 	return dataset.test(FLAGS.data_dir).batch(
	# 		FLAGS.batch_size,drop_remainder=True).make_one_shot_iterator().get_next()

	# Train and evaluate model.
	for _ in range(FLAGS.train_epochs // FLAGS.epochs_between_evals):
		mnist_classifier.train(input_fn=train_input_fn, hooks=None)
		# eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
		# print("\nEvaluation results:\n\t%s\n" % eval_results)


def main(_):
  	run_mnist()


if __name__ == "__main__":
	tf.disable_v2_behavior()
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run()