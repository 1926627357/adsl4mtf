import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import os
import csv

num_classes = [10,1000,10000,65536]
use_fp16 = [0,1]
parallel = ["AUTO_PARALLEL"]
device_num = [1,2,4,8]
subnum = 4

dir1 = "output/output/"
# dir2 = "output/"


def buildfilename(modelname):
    strlist = []
    for a in range(4):
        for b in range(2):
            for c in range(1):
                strtmp = "model_" + modelname + "_num_classes_"+ str(num_classes[a]) + "_use_fp16_" + str(use_fp16[b]) + "_batch_size_32.0_parallel_mode_" + str(parallel[c]) +"_device_num_"
                strlist.append(strtmp)
    return strlist


def getdata(f,dtmp):
    ylen = 4
    d = os.path.join('%s%s' % (dtmp, f))
    ylist = []
    for dnum in device_num:
        ddevice = os.path.join('%s%s' % (d, dnum))
        fdevice = os.path.join('%s%s' % (f, dnum))

        if not os.path.exists(ddevice):
            ylist.append(0)
            ylen -= 1
        else:
            ftmp = re.sub(r'_device_num_\d', '-log.csv', fdevice)
            fcsv = os.path.join('%s/%s' % (ddevice, ftmp))
            with open(fcsv, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                i = 1
                mean_speed = 0
                count = 0
                for row in reader:
                    if i >= 3:
                        mean_speed+=float(row['samples/second'])
                        count+=1
                        # print("[start]")
                        # print(fcsv)
                        # print(ylist)
                        # print()

                    i = i + 1
                ylist.append(mean_speed/count)
    return ylist,ylen

def getLinear(t):
    arr = []
    for i in device_num:
        arr.append(t*i)
    return arr


def gettitle(s):
    num_class = re.findall(r"num_classes_(.+?)_", s)[0]
    use_fp16 = re.findall(r"use_fp16_(.+?)_", s)[0]
    parallel = re.findall(r"parallel_mode_(.+?)", s)[0]
    title = "num_classes:" + num_class + " use_fp16:"+ use_fp16 + " "+ parallel
    return title



def drawbar(modelname):
    plt.figure(modelname, (8, 8))
    fnamelist = buildfilename(modelname)
    pnum = 1

    y_limitation = 1000
    def autolabel(rects):
        for rect in rects:
            height = rect
            height = rect.get_height()
            plt.annotate('{}'.format(int(height)),
                            xy=(rect.get_x()+rect.get_width()/2,height if height<y_limitation-200 else y_limitation/2),
                            xytext=(0,3),
                            textcoords="offset points",
                            ha="center",va="bottom",fontsize=8)

    for subpicname in fnamelist:
        y1 , y1len = getdata(subpicname,dir1)
        # y2 , y2len = getdata(subpicname,dir2)
        linearbar = getLinear(y1[0])
        # if y1len or y2len:
        if y1len:
            plt.subplot(4, 2, pnum)
            plt.subplots_adjust(left=0.06, top=0.92, right=0.96, bottom=0.04, wspace=0.55, hspace=0.55)
            plt.ylabel('samples/second', fontsize=6, labelpad=1)
            plt.xlabel('device_num', fontsize=6, labelpad=0.5)


            plt.title(gettitle(subpicname), fontsize=6)

            # plt.yscale('symlog')
            plt.ylim(0, y_limitation)

            x = list(range(len(y1)))
            total_width, n = 0.8, 2
            width = total_width / n

            # draw 1
            for i in range(len(x)):
                x[i] = x[i] - width
            rects1 = plt.bar(x, y1, width=width, label='mtf-perf', fc='#F62217')
            # draw 2
            # for i in range(len(x)):
            #     x[i] = x[i] + width
            # plt.bar(x, y2, width=width, label='new', tick_label=y1, fc='#D4A017')
            # draw 3
            for i in range(len(x)):
                x[i] = x[i] + width
            rects2 = plt.bar(x, linearbar, width=width, label='o-linear', tick_label=y1, fc='#2B60DE')

            plt.xticks((0, 1, 2, 3), ('1', '2', '4', '8'))
            plt.legend(loc='upper left', fontsize=8)
            pnum += 1
            autolabel(rects1)
            autolabel(rects2)
    plt.suptitle(modelname + "_bactch_size_32_epoch_size_3", fontsize=8)
    plt.savefig("picture/" + modelname + "_bactch_size_32_epoch_size_3.pdf", format="pdf")
   # plt.show()


if __name__ == '__main__':
    # 想要处理的模型的信息就放入到其中
    picname = ["resnet18"]
    for p in picname:
        drawbar(p)
        print(p)
