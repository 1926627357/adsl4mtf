import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf


vgg_dict = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
'''
  input
    |
conv3-64 x m
    |
maxpool stride=2, k=2
    |
conv3-128 x n
    |
maxpool stride=2, k=2
    |
conv3-256 x o
    |
maxpool stride=2, k=2
    |
conv3-512 x p
    |
maxpool stride=2, k=2
    |
conv3-512 x q
    |
maxpool stride=2, k=2
    |

'''
def make_conv_layers(x, mode, batch_norm=True):
    maxpool_count = 0
    conv2d_count = 0
    for size in mode:
        if size == "M":
            x = mtf.layers.max_pool2d(
                                        x,
                                        ksize=(2,2),
                                        name='maxpool'+'-'+str(maxpool_count)
                                        )
            maxpool_count += 1
        else:
            x = mtf.layers.conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name='filters'+'-'+str(conv2d_count),
                                                        size=size
                                                        ),
                            filter_size=(3,3),
                            strides=(1,1),
                            name="conv3"+'-'+str(conv2d_count)
                            )
            
            if batch_norm:
                x = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm"+'-'+str(conv2d_count)
                                )
            x = mtf.relu(x,name="relu"+'-'+str(conv2d_count))
            conv2d_count += 1
    return x

def make_dense_layers(x, classes_dim):
'''
    ...
     |
Dense-4096
     |
Dense-4096
     |
Dense-classnum
'''
    
    dense_dim1 = mtf.Dimension(name="dense_dim1",size=4096)
    dense_dim2 = mtf.Dimension(name="dense_dim2",size=4096)
    x = mtf.layers.dense(x, dense_dim1, name="dense-0")
    x = mtf.layers.dense(x, dense_dim2, name="dense-1")
    x = mtf.layers.dense(x, classes_dim, name="dense-2")
    return x

def VGG(x, classes_dim, depth, batch_norm=True):
    if depth not in vgg_dict.keys():
        print("VGG-{} are not supported!".format(depth))
        raise ValueError
    x = make_conv_layers(x, mode=vgg_dict[depth], batch_norm=batch_norm)

    x = make_dense_layers(x, classes_dim=classes_dim)
    return x