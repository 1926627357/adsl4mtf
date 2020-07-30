import tensorflow.compat.v1 as tf
import os
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
batch_size = 64
image_height = 5
image_width = 5
channels = 1

kh = 1
kw = 1
filters = 3

input = tf.Variable(tf.ones(shape=[batch_size, image_height, image_width, channels]))
filter = tf.Variable(tf.ones(shape=[kh, kw, channels, filters]))
op = tf.nn.conv2d(input, filter, strides=[1, 3, 3, 1], padding='SAME')

init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  print("op:\n",sess.run(op).shape)
print(op)