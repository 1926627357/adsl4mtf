import tensorflow.compat.v1 as tf
white_list = [18,50,101,152]






def resnet_model(numclasses, depth):
    if depth not in white_list:
        print("Renet-{}".format(depth))
        raise ValueError
    else:
        pass
