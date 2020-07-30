import tensorflow.compat.v1 as tf
import os
from config import *
# sess = tf.Session()

image_vec_length = image_height*image_width*num_channels
record_length = 1+image_vec_length

def load_dataset(dataset_root_dir, train_logic=True):
    if train_logic:
        files = [os.path.join(dataset_root_dir,'train','data_batch_{}.bin'.format(i)) for i in range(1,6)]
    dataset = tf.data.FixedLengthRecordDataset(files.pop(0),record_bytes=record_length)
    for otherfile in files:
        tmpdataset = tf.data.FixedLengthRecordDataset(otherfile,record_bytes=record_length)
        dataset = dataset.concatenate(tmpdataset)

    def decode_image(inputstream):
        inputstream = tf.decode_raw(inputstream, tf.uint8)
        image, label = tf.slice(inputstream,[1],[image_vec_length]), tf.slice(inputstream,[0],[1])
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [32*32*3])
        image = image/255
        # label = tf.reshape(label, [])
        # label = tf.to_int32(label)
        return image
    def decode_label(inputstream):
        inputstream = tf.decode_raw(inputstream, tf.uint8)
        image, label = tf.slice(inputstream,[1],[image_vec_length]), tf.slice(inputstream,[0],[1])
        # image = tf.cast(image, tf.float32)
        # image = tf.reshape(image, [32*32*3])
        # image = image/255
        label = tf.reshape(label, [])
        label = tf.to_int32(label)
        return label
    image = dataset.map(decode_image)
    label = dataset.map(decode_label)
    dataset = tf.data.Dataset.zip((image, label))
    return dataset
    
if __name__ == "__main__":
    dataset = load_dataset(dataset_path)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(32)
    dataset = dataset.repeat(10)
    iterator = dataset.make_one_shot_iterator()
    image,lable = iterator.get_next()
    print(image.shape)
    print(lable.shape)