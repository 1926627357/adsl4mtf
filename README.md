# adsl4mtf

## Requirement
our performance script is based on `Tensorflow==2.1`. If you want to know how to install it, please access [tensorflow.install](https://www.tensorflow.org/install)

other requiring packages are all listed in `pip-requirements.txt`, just input:
```
[user@gpu8 /workspace]$ pip install -r adsl4mtf/pip-requirements.txt
```
## USAGE
### directory
1. `adsl4mtf` is the directory of this project
2. `output` used to store log file
3. `picture` used to store pictures 
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
```
[user@gpu8 /workspace]$ python adsl4mtf/run.py\
                                --data_url='/home/haiqwa/dataset/mininet/mini-imagenet-sp2/val'\
                                --num_gpus=4\
                                --models=resnet18,resnet50\
                                --class_nums=10,1000 &> output/xxx.log
```
In modelarts:
|args|value|
|:-:|:-:|
|data_url|/bucket-8048/dataset/mindspore_train/mini-imagenet-sp2/val/|
|num_gpus|8|
|**models**|resnet18,resnet50|
|**class_nums**|10,1000,65536|
|**cloud**||

### log process
```
[user@gpu8 /workspace]$ python adsl4mtf/utils/log_manager.py\
                                --rootDir=output/
```
### draw pictures
```
[user@gpu8 /workspace]$ python adsl4mtf/utils/124gpu.py
```
Before drawing pictures, you should set several configurations in the script
```python
# num_classes and the batch_size in buildfilename should be taken into account
num_classes = [10,1000,10000,65536]
use_fp16 = [0,1]
parallel = ["AUTO_PARALLEL"]
device_num = [1,2,4,8]
subnum = 4
dir1 = "output/output/"
def buildfilename(modelname):
    strlist = []
    for a in range(4):
        for b in range(2):
            for c in range(1):
                strtmp = "model_" + modelname + "_num_classes_"+ str(num_classes[a]) + "_use_fp16_" + str(use_fp16[b]) + "_batch_size_32.0_parallel_mode_" + str(parallel[c]) +"_device_num_"
                strlist.append(strtmp)
    return strlist

...

# config the models
if __name__ == '__main__':
    picname = ["resnet18","resnet50","resnet101","resnet152","vgg11","vgg13","vgg16","vgg19"]
    for p in picname:
        drawbar(p)
        print(p)
```
the output pictures will be stored in `/workspace/picture`

## MeshTF Performance
we measure the performance of mesh tensorflow in vgg and resnet models. And the training configurations are:
|batch size|metric|dataset|precision|class num|
|:-:|:-:|:-:|:-:|:-:|
|32|samples/second|mini-imagenet|FP32|100|

All data below are got from V100 clusters in huawei cloud platform.

|model|1 GPU|2 GPU|4 GPU|8 GPU|
|:-:|:-:|:-:|:-:|:-:|
|vgg11|363|466|806|1180|
|vgg13|242|451|367|672|
|vgg16|205|381|389|606|
|vgg19|179|334|337|550|
|resnet18|469|577|880|1293|
|resnet50|175|281|265|431|
|resnet101|112|160|-|-|
|resent152|79|114|-|-|

auto mixed precision is not supported in mesh tensorflow.
## TODO
* ~~log process programme~~
* wide&deep
* ~~make a dict for mesh shape~~
* ~~fp16 for resnet~~
* ~~fp32 for vgg~~
* ~~mini-dataset upload~~
* ~~replace the cifar10 with the mini-imagenet~~
* mesh shape={'b1':8} -> how to set the value of the class nums for model parallel