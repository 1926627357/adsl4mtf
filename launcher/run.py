import os
import argparse

parser = argparse.ArgumentParser(description="launch the training script")
parser.add_argument("--data_url", required=True,default='/home/haiqwa/dataset/cifar10',help="the bucket path of dataset")
parser.add_argument("--train_url",default=None,help="the output file stored in")
parser.add_argument("--ckpt_path",default=None,help="the root directory path of model checkpoint stored in")
parser.add_argument("--num_gpus",required=True,type=int,default=1,help="the num of devices used to train")

args_opt,_ = parser.parse_known_args()

models = ['resnet18','resnet34','resnet50','resnet101','resnet152','vgg11','vgg13','vgg16','vgg19']


for model in models:
	data_url = args_opt.data_url
	ckpt_path = os.path.join(args_opt.ckpt_path,model)
	epoch = 1
	batch_size = 64
	num_gpus = args_opt.num_gpus
	class_num = 10
	mesh_shape = 'b1:2\\;b2:2'
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