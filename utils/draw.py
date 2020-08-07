
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import os
import csv

xdevice = [1, 2, 4, 8]
subnum = 4
dict1 = {}
pnum = 0

def draw_bar(ytime,ytime2,ytime3, title):
    global pnum
    global dict1
    global lastmodel

    pname = re.findall(r"model_(.+?)_",title)[0]
    title1 = "picture//" + pname
    if (lastmodel != pname):
        if( pnum > 0):
            plt.suptitle(lastmodel+ "_bactch_size_32_epoch_size_3",fontsize = 8)
            plt.savefig("picture/"+ lastmodel+ "_bactch_size_32_epoch_size_3",format="pdf")
            plt.show()
            pnum = 0
            lastmodel = pname
            plt.figure(pname,(8,8))
    if (pnum == 0):
        plt.figure(pname, (8, 8))
        lastmodel = pname


    plt.subplot(subnum, subnum, pnum % 16 + 1)
    plt.subplots_adjust(left=0.06, top=0.92, right=0.96, bottom=0.04, wspace=0.55, hspace=0.55)
    plt.ylabel('samples/second',fontsize=6,labelpad=1)
    plt.xlabel('device_num',fontsize=6,labelpad=0.5)
    plt.title(gettitle(title),fontsize=6)


    plt.yscale('symlog')
    plt.ylim(0, 100000)

    x = list(range(len(ytime)))
    total_width, n = 0.4, 3
    width = total_width / n

    #draw 1
    for i in range(len(x)):
        x[i] = x[i] - width
    plt.bar(x, ytime, width=width, label='old', fc='#F62217')
    #draw 2
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, ytime2, width=width, label='new', tick_label=ytime, fc='#D4A017')
    #draw 3
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, ytime3, width=width, label='o-linear', tick_label=ytime, fc='#2B60DE')

    plt.xticks((0, 1, 2, 3), ('1', '2', '4', '8'))
    plt.legend(loc='upper left',fontsize=4)



def gettitle(s):
    num_class = re.findall(r"num_classes_(.+?)_", s)[0]
    use_fp16 = re.findall(r"use_fp16_(.+?)_", s)[0]
    parallel = re.findall(r"parallel_mode_(.+?)_epoch", s)[0]
    title = "num_classes:" + num_class + " use_fp16:"+ use_fp16 + " "+ parallel
    return title


def getLinear(t):
    arr = []
    for i in xdevice:
        arr.append(t*i)
    return arr



def getdata(dname, fname):
    d = dname[:-1]
    f = fname[:-1]

    global pnum
    global dict1
    global lastmodel

    ytime = []
    for dnum in xdevice:
        ddevice = os.path.join('%s%s' % (d, dnum))
        fdevice = os.path.join('%s%s' % (f, dnum))

        dict1[fdevice] = 1
        if not os.path.exists(ddevice):
            ytime.append(0)
        else:
            ftmp = re.sub(r'_device_num_\d', '-log.csv', fdevice)
            fcsv = os.path.join('%s/%s' % (ddevice, ftmp))
            with open(fcsv, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                i=1
                for row in reader:
                    if i == 3:
                        ytime.append(float(row['samples/second']))
                    i = i + 1

    return ytime

if __name__ == '__main__':

    global lastmodel

    dirname_old = "output_old/"
    dirname_new = "output_old/"
    files = os.listdir(dirname_old)
    filestmp = os.listdir(dirname_new)
    for f in filestmp:
        files.append(f)
    sorted(set(files))
    #files = list(files)
    print(files)

    tt = 0
    for f in files:
        dict1[f] = 0
        if(tt == 0):
            lastmodel = re.findall(r"model_(.+?)_",f)[0]
            tt = 1

    for fname in files:
        dname_old = os.path.join('%s%s' % (dirname_old, fname))
        dname_new = os.path.join('%s%s' % (dirname_new, fname))
        if(dict1[fname] == 0 ):
            bar1 = getdata(dname_old,fname)
            bar2 = getdata(dname_new,fname)
            linearbar = getLinear(bar1[0])

            draw_bar(bar1, bar2,linearbar, fname[:-13])

            pnum = pnum + 1

    plt.suptitle(lastmodel + "_bactch_size_32_epoch_size_3", fontsize=8)
    plt.savefig("picture/" + lastmodel + "_bactch_size_32_epoch_size_3")
    plt.show()
