import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
'''
ERROR: 3
WARNING: 2
INFO: 1
DEBUG: 0
'''
os.environ['GLOG_v'] = '0'
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
sys.path.append(os.path.abspath('.'))


import mesh_tensorflow as mtf
from adsl4mtf.dataset import ctr_engine  # local file import
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity (tf.logging.INFO)


from adsl4mtf.model_zoo import network
import mesh_tensorflow.auto_mtf
from mesh_tensorflow.auto_mtf import layout_optimizer
from mesh_tensorflow.auto_mtf import memory_estimator
# disable the eager graph mode in tf2.1
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

# get user input
import argparse
parser = argparse.ArgumentParser(description='adsl mesh tensorflow performance benchmark script')
parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
# parser.add_argument('--log_url', default=None, help='The path of the log')
parser.add_argument('--ckpt_path', default=None, help='Location of the model parameter checkpoint')
parser.add_argument('--model', required=True, default='resnet50', help='the neural network used to train')
parser.add_argument('--epoch', type=int, default=5, help='Train epoch size.')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of the input stream')
parser.add_argument('--num_gpus', type=int, default=4, help='the number of devices used to train model')
parser.add_argument('--class_num', type=int, default=10, help='the classes num of label in dataset')
parser.add_argument('--mesh_shape', default="b1:2;b2:2", help='the shape of the devices, like: \"b1:2;b2:2\"')
parser.add_argument('--fp16', action='store_true', help='decide to use fp16 or not')
# parser.add_argument('--cloud', action='store_true', help='training in cloud or not')
args_opt, unknown = parser.parse_known_args()



import adsl4mtf.log as logger

def model_backbone(id_hldr, wt_hldr, labels, mesh):
	"""The model.
	Args:
		image: tf.Tensor with shape [batch, 32*32]
		labels: a tf.Tensor with shape [batch] and dtype tf.int32
		mesh: a mtf.Mesh
	Returns:
		logits: a mtf.Tensor with shape [batch, 10]
		loss: a mtf.Tensor with shape []
	"""


	batch_dim = mtf.Dimension("batch",args_opt.batch_size)
    field_dim = mtf.Dimension("field",39)
    vocab_dim = mtf.Dimension("vocab_size",20000)
    embed_dim = mtf.Dimension("embed_size",80)
	outdim = mtf.Dimension("outdim",1)
	id_hldr = mtf.import_tf_tensor(
		mesh, tf.reshape(id_hldr, [args_opt.batch_size, field_dim.size]),
		mtf.Shape(
			[batch_dim, field_dim]))
	wt_hldr = mtf.import_tf_tensor(
		mesh, tf.reshape(wt_hldr, [args_opt.batch_size, field_dim.size]),
		mtf.Shape(
			[batch_dim, field_dim]))
	if args_opt.fp16:
		float16=mtf.VariableDType(tf.float16,tf.float16,tf.float16)
		# id_hldr=mtf.cast(id_hldr,dtype=tf.int32)
        wt_hldr=mtf.cast(wt_hldr,dtype=tf.float16)
	else:
		float16=None

	logits = network[args_opt.model](id_hldr, wt_hldr, vocab_dim, embed_dim, outdim,float16=float16)
	logits = mtf.cast(logits,dtype=tf.float32)

	if labels is None:
		loss = None
	else:
		labels = mtf.import_tf_tensor(
			mesh, tf.reshape(labels, [args_opt.batch_size]), mtf.Shape([batch_dim]))
		loss = mtf.layers.sigmoid_cross_entropy_with_logits(logits,label)
		loss = mtf.reduce_mean(loss)
	return logits, loss

def model_fn(features, labels, mode, params):
	"""The model_fn argument for creating an Estimator."""
	global_step = tf.train.get_global_step()
	graph = mtf.Graph()
	mesh = mtf.Mesh(graph, "my_mesh")
	logits, loss = model_backbone(features, labels, mesh)
	
	variables = graph._all_variables
	for v in variables:
		logger.debug("[parameter] (name,shape,dtype): ({},{},{})".format(v.name,v.shape,v.dtype.master_dtype))
	mesh_shape = mtf.convert_to_shape(args_opt.mesh_shape)
	# layout_rules = mtf.auto_mtf.layout(graph, mesh_shape, [logits, loss])
	mesh_shape = mtf.convert_to_shape(mesh_shape)
	estimator = memory_estimator.MemoryEstimator(graph, mesh_shape, [logits, loss])
	optimizer = layout_optimizer.LayoutOptimizer(estimator,scheduler_alg="NAIVE")
	layout_rules =  mtf.convert_to_layout_rules(optimizer.solve())



	logger.info("[auto mtf search] strategy: {}".format(layout_rules))
	mesh_devices = ["gpu:{}".format(i) for i in range(int(args_opt.num_gpus))]
	mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape, layout_rules, mesh_devices)



	if mode == tf.estimator.ModeKeys.TRAIN:
		var_grads = mtf.gradients(
			[loss], [v.outputs[0] for v in graph.trainable_variables])
		optimizer = mtf.optimize.SgdOptimizer(0.01)
		# optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
		update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)

	lowering = mtf.Lowering(graph, {mesh: mesh_impl})
	restore_hook = mtf.MtfRestoreHook(lowering)

	tf_logits = lowering.export_to_tf_tensor(logits)
	if mode != tf.estimator.ModeKeys.PREDICT:
		tf_loss = lowering.export_to_tf_tensor(loss)

	if mode == tf.estimator.ModeKeys.TRAIN:
		tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
		tf_update_ops.append(tf.assign_add(global_step, 1))
		train_op = tf.group(tf_update_ops)

		accuracy = tf.metrics.accuracy(
			labels=labels, predictions=tf.argmax(tf_logits, axis=1))

		# Name tensors to be logged with LoggingTensorHook.
		tf.identity(tf_loss, "cross_entropy")
		tf.identity(accuracy[1], name="train_accuracy")

		logging_hook = tf.train.LoggingTensorHook(every_n_iter=100,tensors={'loss': 'cross_entropy','acc':'train_accuracy'})

		# restore_hook must come before saver_hook
		return tf.estimator.EstimatorSpec(
			tf.estimator.ModeKeys.TRAIN, loss=tf_loss, train_op=train_op,
			training_chief_hooks=[restore_hook,logging_hook])

def run():
	"""Run model training loop."""
	mnist_classifier = tf.estimator.Estimator(
		model_fn=model_fn,
		model_dir=args_opt.ckpt_path)

	# Set up training and evaluation input functions.
	def train_input_fn():
		"""Prepare data for training."""
		# When choosing shuffle buffer sizes, larger sizes result in better
		# randomness, while smaller sizes use less memory. MNIST is a small
		# enough dataset that we can easily shuffle the full epoch.
		
		# ds = load_dataset(args_opt.data_url,use_fp16=args_opt.fp16)
		ds = ctr_engine(RootDir=args_opt.data_url)
		ds_batched = ds.cache().shuffle(buffer_size=args_opt.batch_size*2).batch(args_opt.batch_size,drop_remainder=True)

		# Iterate through the dataset a set number (`epochs_between_evals`) of times
		# during each training session.
		ds = ds_batched.repeat(args_opt.epoch)
		return ds

	mnist_classifier.train(input_fn=train_input_fn, hooks=None)

logger.info("=======================================[BEGINE]=======================================")
logger.info('[Input args] {}'\
				.format(
							{
								'data_url':args_opt.data_url,
								'ckpt_path':args_opt.ckpt_path,
								'model':args_opt.model,
								'epoch':args_opt.epoch,
								'batch_size':args_opt.batch_size,
								'num_gpus':args_opt.num_gpus,
								'class_num':args_opt.class_num,
								'mesh_shape':args_opt.mesh_shape,
								'fp16':str(args_opt.fp16)}
							)
			)
logger.info('[configuration]model_{}_num_classes_{}_use_fp16_{}_batch_size_{}_parallel_mode_AUTO_PARALLEL_epoch_size_{}_device_num_{}'\
				.format(
					args_opt.model,
					args_opt.class_num,
					1 if args_opt.fp16 else 0,
					args_opt.batch_size/args_opt.num_gpus,
					args_opt.epoch,
					args_opt.num_gpus
				))
run()