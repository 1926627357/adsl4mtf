import os
import argparse
# '/home/haiqwa/dataset/criteo/tfrecord/train.tfrecord'
# '/home/haiqwa/dataset/mininet/mini-imagenet-sp2/val'
parser = argparse.ArgumentParser(description="launch the training script")
parser.add_argument("--data_url",default='/home/haiqwa/dataset/mininet/mini-imagenet-sp2/val',help="the bucket path of dataset")
parser.add_argument("--train_url",default=None,help="the output file stored in")
parser.add_argument("--ckpt_path",default='./ckpt',help="the root directory path of model checkpoint stored in")
parser.add_argument("--num_gpus",type=int,default=1,help="the num of devices used to train")
parser.add_argument('--cloud', action='store_true', help='training in cloud or not')
args_opt,_ = parser.parse_known_args()

# copy data using moxing
if args_opt.cloud:
	import moxing as mox
	local_data_path = './data'
	mox.file.copy_parallel(src_url=args_opt.data_url, dst_url=local_data_path)
else:
	local_data_path = args_opt.data_url


models = ['vgg11']
class_nums = [100]
fp16Choices = [True]
meshShapeDict={
	1:['b1:1'],
	2:['b1:2'],
	4:['b1:2\\;b2:2'],
	8:['b1:2\\;b2:4']
}



for model in models:
	for class_num in class_nums:
		for fp16 in fp16Choices:
			for mesh_shape in meshShapeDict[args_opt.num_gpus]:
				launch_name = 'WDlaunch' if model=='widedeep' else 'CVlaunch'
				data_url = local_data_path
				ckpt_path = os.path.join(args_opt.ckpt_path,model,str(class_num),'1' if fp16 else '0',str(len(mesh_shape)))
				
				epoch = 5
				batch_size = 32*args_opt.num_gpus
				num_gpus = args_opt.num_gpus
				# class_num = 10
				# mesh_shape = 'b1:2\\;b2:2'
				# mesh_shape = meshShapeDict[num_gpus]
				cmd = "python adsl4mtf/launcher/{}.py \
							--data_url={} \
							--ckpt_path={} \
							--model={} \
							--epoch={} \
							--batch_size={} \
							--num_gpus={} \
							--class_num={} \
							--mesh_shape={} \
							{}".format(
								launch_name,
								data_url,
								ckpt_path,
								model,
								epoch,
								batch_size,
								num_gpus,
								class_num,
								mesh_shape,
								'--fp16' if fp16 else ''
							)
				os.system(cmd)
# print(cmd)