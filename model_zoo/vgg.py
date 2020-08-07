import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf
import adsl4mtf.log as logger
from adsl4mtf.patch.conv2d import conv2d
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
def make_conv_layers(x, mode, batch_norm=True, float16=None):
    maxpool_count = 0
    conv2d_count = 0
    for size in mode:
        if size == "M":
            x = mtf.layers.max_pool2d(
                                        x,
                                        ksize=(2,2),
                                        name='maxpool'+'-'+str(maxpool_count)
                                        )
            logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
            maxpool_count += 1
        else:
            x = conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name='filters'+'-'+str(conv2d_count),
                                                        size=size
                                                        ),
                            filter_size=(3,3),
                            strides=(1,1),
                            name="conv3"+'-'+str(conv2d_count),
                            variable_dtype=float16
                            )
            logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
            if batch_norm:
                x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm"+'-'+str(conv2d_count)
                                )
                logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
            x = mtf.relu(x,name="relu-conv"+'-'+str(conv2d_count))
            logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
            conv2d_count += 1
    return x
'''
    ...
     |
Dense-4096
     |
Dense-4096
     |
Dense-classnum
'''
def make_dense_layers(x, classes_dim, float16=None):
    
    
    dense_dim1 = mtf.Dimension(name="dense_dim1",size=4096)
    dense_dim2 = mtf.Dimension(name="dense_dim2",size=4096)
    x = mtf.layers.dense(x, dense_dim1, name="dense-0",variable_dtype=float16)
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu-dense-0")
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.layers.dense(x, dense_dim2, name="dense-1",variable_dtype=float16)
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu-dense-1")
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.layers.dense(x, classes_dim, name="dense-2",variable_dtype=float16)
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    return x

def vgg(x, classes_dim, depth, batch_norm=False, float16=None):
    if float16:
        x = mtf.cast(x,dtype=tf.float16)
    logger.debug("[input tensor] (name,shape):({},{})".format(x.name,x.shape))
    if depth not in vgg_dict.keys():
        logger.error("VGG{} are not supported".format(depth))
        raise ValueError
    x = make_conv_layers(x, mode=vgg_dict[depth], batch_norm=batch_norm,float16=float16)

    x = mtf.reshape(
                        x, 
                        new_shape=[
                                    x.shape.dims[0],
                                    mtf.Dimension(
                                        name="flatten",
                                        size=x.shape.dims[1].size*x.shape.dims[2].size*x.shape.dims[3].size
                                    )
                                    ],
                        name="flatten"
                        )
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = make_dense_layers(x, classes_dim=classes_dim,float16=float16)
    
    return x


def vgg11(x, classes_dim, batch_norm=False,float16=None):
    x = vgg(x=x,classes_dim=classes_dim,depth=11,batch_norm=batch_norm,float16=float16)
    return x

def vgg13(x, classes_dim, batch_norm=False,float16=None):
    x = vgg(x=x,classes_dim=classes_dim,depth=13,batch_norm=batch_norm,float16=float16)
    return x

def vgg16(x, classes_dim, batch_norm=False,float16=None):
    x = vgg(x=x,classes_dim=classes_dim,depth=16,batch_norm=batch_norm,float16=float16)
    return x

def vgg19(x, classes_dim, batch_norm=True,float16=None):
    x = vgg(x=x,classes_dim=classes_dim,depth=19,batch_norm=batch_norm,float16=float16)
    return x