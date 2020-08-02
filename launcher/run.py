import os
import argparse

parser = argparse.ArgumentParser(description="launch the training script")
parser.add_argument("--data_url",default='/home/haiqwa/dataset/cifar10',help="the bucket path of dataset")
parser.add_argument("--train_url",default=None,help="the output file stored in")
parser.add_argument("--ckpt_path",default='./ckpt',help="the root directory path of model checkpoint stored in")
parser.add_argument("--num_gpus",required=True,type=int,default=1,help="the num of devices used to train")

args_opt,_ = parser.parse_known_args()

models = ['vgg11','vgg13','vgg16','vgg19']
class_nums = [10,1024,65536,65536*2]

for model in models:
	for class_num in class_nums:
		data_url = args_opt.data_url
		ckpt_path = os.path.join(os.path.join(args_opt.ckpt_path,model),str(class_num))
		epoch = 1
		batch_size = 64
		num_gpus = args_opt.num_gpus
		# class_num = 10
		# mesh_shape = 'b1:2\\;b2:2'
		mesh_shape = 'b1:2'
		cmd = "python adsl4mtf/launcher/main.py \
					--data_url={} \
					--ckpt_path={} \
					--model={} \
					--epoch={} \
					--batch_size={} \
					--num_gpus={} \
					--class_num={} \
					--mesh_shape={} ".format(
						data_url,
						ckpt_path,
						model,
						epoch,
						batch_size,
						num_gpus,
						class_num,
						mesh_shape
					)
		os.system(cmd)
# print(cmd)