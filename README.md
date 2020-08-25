# adsl4mtf

## TODO
* ~~log process programme~~
* wide&deep
* ~~make a dict for mesh shape~~
* ~~fp16 for resnet~~
* ~~fp32 for vgg~~
* ~~mini-dataset upload~~
* ~~replace the cifar10 with the mini-imagenet~~
* mesh shape={'b1':8} -> how to set the value of the class nums for model parallel

## USAGE
### directory
```
/workspace
|
|--- adsl4mtf
|
|--- output
|      |
|      |--- output
|      |
|      |--- xxx.log
|
|--- picture
```
### run script
In server:
> \[user@gpu8 /workspace\]$ python adsl4mtf/run.py\\
>                                           --data_url='/home/haiqwa/dataset/mininet/mini-imagenet-sp2/val'\\
>                                           --num_gpus=4\\
>                                           --models=resnet18,resnet50\\
>                                           --class_nums=10,1000 &> output/xxx.log

In modelarts:
|args|value|
|:-:|:-:|
|data_url|/bucket-8048/dataset/mindspore_train/mini-imagenet-sp2/val/|
|num_gpus|8|
|**models**|resnet18,resnet50|
|**class_nums**|10,1000,65536|
|**cloud**||

### log process
> \[user@gpu8 /workspace\]$ python adsl4mtf/utils/log_manager.py\\
>                                                   --rootDir=output/

### draw pictures
> \[user@gpu8 /workspace\]$ python adsl4mtf/utils/124gpu.py
