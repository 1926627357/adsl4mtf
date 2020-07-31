import os
import argparse

parser = argparse.ArgumentParser(description="launch the training script")
parser.add_argument("--data_url", required=True,default=None,help="the bucket path of dataset")
parser.add_argument("--train_url",default=None,help="the output file stored in")
parser.add_argument("--num_gpus",required=True,type=int,default=1,help="the num of devices used to train")

args_opt,_ = parser.parse_known_args()

print("="*10,1,"="*10)
# os.system("python adsl4mtf/launcher/main.py \
#             --data_url={}\
#         --model=resnet50 \
#         --epoch=1 \
#         --batch_size=64 \
#         --device_num={} \
#         --class_num=10 \
#         --mesh_shape=b1:2\\;b2:2 ".format(args_opt.data_url,args_opt.num_gpus))
print("python adsl4mtf/launcher/main.py \
        --data_url={}\
        --ckpt_path={}\
        --model=resnet50 \
        --epoch=1 \
        --batch_size=64 \
        --device_num={} \
        --class_num=10 \
        --mesh_shape=b1:2\\;b2:2 ".format(args_opt.data_url,args_opt.ckpt_path,args_opt.num_gpus))