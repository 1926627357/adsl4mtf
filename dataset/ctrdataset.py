import os
import random
import numpy as np
import tensorflow.compat.v1 as tf
image_feature_description={
        'label': tf.io.FixedLenFeature([], tf.float32),
        'ids': tf.io.FixedLenFeature([], tf.string),
        'wts': tf.io.FixedLenFeature([], tf.string)
}
def _parse_image_function(example_proto):
  input_dict = tf.io.parse_single_example(example_proto, image_feature_description)
  return    (tf.io.decode_raw(input_dict['ids'], out_type=tf.int32),\
            tf.io.decode_raw(input_dict['wts'], out_type=tf.float32)),\
            input_dict['label']
def ctr_engine(filepath):
    # the input file is in tfrecord format
    
    
    
    dataset = tf.data.TFRecordDataset(os.path.join(filepath,'train.tfrecord'))
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = dataset.map(_parse_image_function, num_parallel_calls=AUTOTUNE)

    dataset = dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
if __name__ == "__main__":
    # tf.disable_v2_behavior()
    tfrecord_filename = '/home/haiqwa/dataset/criteo/tfrecord/train.tfrecord'
    db_instance = ctr_engine(tfrecord_filename)
    db_instance = db_instance.batch(16000)
    iterator = db_instance.make_one_shot_iterator()
    (ids,wts),label = next(iter(db_instance))
    print("===ids===")
    print(ids)
    print("===wts===")
    print(wts)
    print("===label===")
    print(label)
    # import time
    # with tf.Session() as sess:
    #     ids,wts,label = sess.run(iterator.get_next())
    #     t1=time.time()
    #     ids,wts,label = sess.run(iterator.get_next())
    #     ids,wts,label = sess.run(iterator.get_next())
    #     ids,wts,label = sess.run(iterator.get_next())
    #     t2=time.time()
    #     print("cost time per step",(t2-t1)/3)