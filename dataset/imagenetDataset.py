from PIL import Image
import os
import random
import numpy as np
import tensorflow.compat.v1 as tf
import pathlib
# imagepath='/home/haiqwa/dataset/mininet/mini-imagenet-sp1/train/n07697537/n0153282900000005.jpg'
# RootDir = '/home/haiqwa/dataset/mininet/mini-imagenet-sp1/train/'

def read_image(images_folder):
	""" It reads a single image file into a numpy array and preprocess it
		Args:
			images_folder: path where to random choose an image
		Returns:
			im_array: the numpy array of the image [width, height, channels]
	"""
	# random image choice inside the folder 
	# (randomly choose an image inside the folder)
	image_path = os.path.join(images_folder, random.choice(os.listdir(images_folder)))
	
	# load and normalize image
	im_array = preprocess_image(image_path)
	#im_array = read_k_patches(image_path, 1)[0]
		
	return im_array

def preprocess_image(image_path):
	""" It reads an image, it resize it to have the lowest dimesnion of 256px,
		it randomly choose a 224x224 crop inside the resized image and normilize the numpy 
		array subtracting the ImageNet training set mean
		Args:
			images_path: path of the image
		Returns:
			cropped_im_array: the numpy array of the image normalized [width, height, channels]
	"""
	image_path=str(image_path,encoding='utf-8')
	IMAGENET_MEAN = [123.68, 116.779, 103.939] # rgb format

	img = Image.open(image_path).convert('RGB')

	# resize of the image (setting lowest dimension to 256px)
	if img.size[0] < img.size[1]:
		h = int(float(256 * img.size[1]) / img.size[0])
		img = img.resize((256, h), Image.ANTIALIAS)
	else:
		w = int(float(256 * img.size[0]) / img.size[1])
		img = img.resize((w, 256), Image.ANTIALIAS)

	# random 244x224 patch
	x = random.randint(0, img.size[0] - 224)
	y = random.randint(0, img.size[1] - 224)
	img_cropped = img.crop((x, y, x + 224, y + 224))

	cropped_im_array = np.array(img_cropped, dtype=np.float32)

	for i in range(3):
		cropped_im_array[:,:,i] -= IMAGENET_MEAN[i]

	#for i in range(3):
	#	mean = np.mean(img_c1_np[:,:,i])
	#	stddev = np.std(img_c1_np[:,:,i])
	#	img_c1_np[:,:,i] -= mean
	#	img_c1_np[:,:,i] /= stddev

	return cropped_im_array



def create_dataset(RootDir):
	RootDir = pathlib.Path(RootDir)
	dataset = tf.data.Dataset.list_files(str(RootDir/'*/*'), shuffle=False)
	class_names = sorted([item.name for item in RootDir.glob('*')])
	
	def get_label(imagepath):
		parts = str(imagepath,encoding='utf-8').split(os.path.sep)
		label = class_names.index(parts[-2])
		return tf.convert_to_tensor(label)

	def process_path(imagepath):
		label = get_label(imagepath)
		image = preprocess_image(imagepath)
		return image,label
	dataset=dataset.map(process_path,num_parallel_calls=AUTOTUNE)
	return dataset
	# print(get_label(b'/home/haiqwa/dataset/mininet/mini-imagenet-sp1/train/n01532829/n0153282900000005.jpg'))
		
# tf.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()
# create_dataset(RootDir=RootDir)
# image = read_image('/home/haiqwa/dataset/mininet/mini-imagenet-sp1/train/n07697537/')
# print(image.shape)