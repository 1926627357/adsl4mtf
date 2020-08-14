import os
import random
import numpy as np
import tensorflow.compat.v1 as tf

def ctr_engine(filepath):
    # the input file is in tfrecord format
    def stream_decoder(example_proto):
        input_dict = tf.io.parse_single_example(example_proto, image_feature_description)
        return    tf.io.decode_raw(input_dict['ids'], out_type=tf.int32),\
                    tf.io.decode_raw(input_dict['wts'], out_type=tf.float32),\
                    input_dict['label']
    dataset = tf.data.TFRecordDataset(filepath)
    image_feature_description={
        'label': tf.io.FixedLenFeature([], tf.float32),
        'ids': tf.io.FixedLenFeature([], tf.string),
        'wts': tf.io.FixedLenFeature([], tf.string)
    }
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = dataset.map(stream_decoder, num_parallel_calls=AUTOTUNE)

    dataset = dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
if __name__ == "__main__":
    tfrecord_filename = '/home/haiqwa/dataset/criteo/tfrecord/train.tfrecord'
    db_instance = ctr_engine(ctr_engine)
    db_instance = db_instance.batch(10)
    # iterator = dataset.make_one_shot_iterator()
    ids,wts,label = next(iter(db_instance))
    print("===ids===")
    print(ids)
    print("===wts===")
    print(wts)
    print("===label===")
    print(label)