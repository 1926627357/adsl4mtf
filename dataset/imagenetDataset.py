import os
import random
import numpy as np
import tensorflow.compat.v1 as tf
import pathlib
# imagepath='/home/haiqwa/dataset/mininet/mini-imagenet-sp1/train/n07697537/n0153282900000005.jpg'


def imagenet_engine(RootDir):
	RootDir = pathlib.Path(RootDir)
	all_image_paths = list(RootDir.glob('*/*'))
	all_image_paths = [str(path) for path in all_image_paths]
	random.shuffle(all_image_paths)

	# 38400张图片
	print("[image num]: ",len(all_image_paths))

	# dataset = tf.data.Dataset.list_files(str(RootDir/'*/*'), shuffle=False)
	class_names = sorted([item.name for item in RootDir.glob('*')])
	print("[class num]: ",len(class_names))
	class_dict = {class_name:index for index,class_name in enumerate(class_names)}
	all_image_labels = [class_dict[pathlib.Path(path).parent.name]
                    		for path in all_image_paths]
	dataset_image = tf.data.Dataset.from_tensor_slices(all_image_paths)
	dataset_label = tf.data.Dataset.from_tensor_slices(all_image_labels)
	def load_and_preprocess_image(path):
  		image = tf.read_file(path)
  		return preprocess_image(image)
	def preprocess_image(image):
  		image = tf.image.decode_jpeg(image, channels=3)
  		image = tf.image.resize(image, [224, 224])
  		image /= 255.0  # normalize to [0,1] range
  		return image
	AUTOTUNE = tf.data.experimental.AUTOTUNE
	dataset_image = dataset_image.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
	dataset = tf.data.Dataset.zip((dataset_image, dataset_label))
	dataset = dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)
	return dataset
		
# tf.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()
# image = read_image('/home/haiqwa/dataset/mininet/mini-imagenet-sp1/train/n07697537/')
# print(image.shape)

# image = tf.read_file('/home/haiqwa/dataset/mininet/mini-imagenet-sp1/train/n07697537/n0769753700001159.jpg')
# print(image)
if __name__ == "__main__":
	RootDir = '/home/haiqwa/dataset/mininet/mini-imagenet-sp2/val/'
	dataset = imagenet_engine(RootDir=RootDir)
	dataset = dataset.batch(64)
	import time
	# t1=time.time()
	image,label = next(iter(dataset))
	t1=time.time()
	image,label = next(iter(dataset))
	image,label = next(iter(dataset))
	image,label = next(iter(dataset))
	t2=time.time()
	print(t2-t1)
	print(image.shape)
	
