import tensorflow.compat.v1 as tf
tfrecord_filename = '/home/haiqwa/dataset/criteo/tfrecord/train.tfrecord'
tf.disable_v2_behavior()
def _parse_image_function(example_proto):
  input_dict = tf.io.parse_single_example(example_proto, image_feature_description)
  return    tf.io.decode_raw(input_dict['ids'], out_type=tf.int32),\
            tf.io.decode_raw(input_dict['wts'], out_type=tf.float32),\
            input_dict['label']


dataset = tf.data.TFRecordDataset(tfrecord_filename)
image_feature_description={
    'label': tf.io.FixedLenFeature([], tf.float32),
    'ids': tf.io.FixedLenFeature([], tf.string),
    'wts': tf.io.FixedLenFeature([], tf.string)
}
AUTOTUNE = tf.data.experimental.AUTOTUNE
dataset=dataset.map(_parse_image_function, num_parallel_calls=AUTOTUNE)

dataset = dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)
iterator = dataset.make_one_shot_iterator()
index = 0


dataset = dataset.batch(16000)
import time
# ids,wts,label = next(iter(dataset))
# t1=time.time()
# ids,wts,label = next(iter(dataset))
# ids,wts,label = next(iter(dataset))
# ids,wts,label = next(iter(dataset))
# t2=time.time()
# print("cost time per step",(t2-t1)/3)

with tf.Session() as sess:
	t1=time.time()
	ids,wts,label = sess.run(iterator.get_next())
	ids,wts,label = sess.run(iterator.get_next())
	ids,wts,label = sess.run(iterator.get_next())
	t2=time.time()
	print("cost time per step",(t2-t1)/3)