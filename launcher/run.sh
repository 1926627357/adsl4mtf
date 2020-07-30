python /home/haiqwa/document/test/adsl4mtf/launcher/main.py\
        --data_url=/home/haiqwa/dataset/cifar10\
        --model=resnet50\ 
        --epoch=1\ 
        --batch_size=64\
        --device_num=4\ 
        --class_num=10\ 
        --mesh_shape=b1:2;b2:2\ 
        &>> /home/haiqwa/document/test/adsl4mtf/launcher/log