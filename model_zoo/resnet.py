import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf
import adsl4mtf.log as logger
from adsl4mtf.patch.conv2d import conv2d
white_list = [18,34,50,101,152]



def ResidualBlockWithDown(x, order, out_channels, strides,float16=None,batch_norm=False):
    name = "ResidualBlockWithDown"
    expansion = 4
    out_chls = out_channels // expansion
    identity = x

    x = conv2d(
                            x, 
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters1',
                                                        size=out_chls
                                                        ),
                            filter_size=(1,1),
                            strides=(1,1),
                            name="conv1x1_RBW_1"+'-'+str(order),
                            variable_dtype=float16
                            )
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    if batch_norm:
        x,_ = mtf.layers.batch_norm(
                                    x,
                                    is_training=True,
                                    momentum=0.99,
                                    epsilon=1e-5,
                                    name="batch_norm_RBW_1"+'-'+str(order)
                                    )
        logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_RBW_1"+'-'+str(order))
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters2',
                                                        size=out_chls
                                                        ),
                            filter_size=(3,3),
                            strides=strides,
                            name="conv3x3_RBW_1"+'-'+str(order),
                            variable_dtype=float16
                            )
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    if batch_norm:
        x,_ = mtf.layers.batch_norm(
                                    x,
                                    is_training=True,
                                    momentum=0.99,
                                    epsilon=1e-5,
                                    name="batch_norm_RBW_2"+'-'+str(order)
                                    )

        logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_RBW_2"+'-'+str(order))
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters3',
                                                        size=out_channels
                                                        ),
                            filter_size=(1,1),
                            strides=(1,1),
                            name="conv1x1-2_RBW_2"+'-'+str(order),
                            variable_dtype=float16
                            )
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    if batch_norm:
        x,_ = mtf.layers.batch_norm(
                                    x,
                                    is_training=True,
                                    momentum=0.99,
                                    epsilon=1e-5,
                                    name="batch_norm_RBW_3"+'-'+str(order)
                                    )
        logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    identity = conv2d(
                                    identity,
                                    output_dim=mtf.Dimension(
                                                                name=name+'-'+str(order)+'-'+'filters3',
                                                                size=out_channels
                                                                ),
                                    filter_size=(1,1),
                                    strides=strides,
                                    name="conv1x1_RBW_3"+'-'+str(order),
                                    variable_dtype=float16
                                    )
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    if batch_norm:
        identity,_ = mtf.layers.batch_norm(
                                            identity,
                                            is_training=True,
                                            momentum=0.99,
                                            epsilon=1e-5,
                                            name="batch_norm_RBW_4"+'-'+str(order)
                                            )
        logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    identity = mtf.reshape(
                            identity, 
                            new_shape=[identity.shape.dims[0],identity.shape.dims[1],identity.shape.dims[2], x.shape.dims[3]],
                            name="reshape_RBW"+str(order)
                            )
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.add(x,identity,output_shape=x.shape,name="add_RBW_1"+'-'+str(order))
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_RBW_3"+'-'+str(order))
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    return x

def ResidualBlock(x, order, out_channels, strides,float16=None,batch_norm=False):
    name = "ResidualBlock"
    expansion = 4
    out_chls = out_channels // expansion
    identity = x

    x = conv2d(
                            x, 
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters1',
                                                        size=out_chls
                                                        ),
                            filter_size=(1,1),
                            strides=(1,1),
                            name="conv1x1_RB_1"+'-'+str(order),
                            variable_dtype=float16
                            )
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    if batch_norm:
        x,_ = mtf.layers.batch_norm(
                                    x,
                                    is_training=True,
                                    momentum=0.99,
                                    epsilon=1e-5,
                                    name="batch_norm_RB_1"+'-'+str(order)
                                    )
        logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_RB_1"+'-'+str(order))
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters2',
                                                        size=out_chls
                                                        ),
                            filter_size=(3,3),
                            strides=strides,
                            name="conv3x3_RB_1"+'-'+str(order),
                            variable_dtype=float16
                            )
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    if batch_norm:
        x,_ = mtf.layers.batch_norm(
                                    x,
                                    is_training=True,
                                    momentum=0.99,
                                    epsilon=1e-5,
                                    name="batch_norm_RB_2"+'-'+str(order)
                                    )
        logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_RB_2"+'-'+str(order))
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters3',
                                                        size=out_channels
                                                        ),
                            filter_size=(1,1),
                            strides=(1,1),
                            name="conv1x1_RB_2"+'-'+str(order),
                            variable_dtype=float16
                            )
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    if batch_norm:
        x,_ = mtf.layers.batch_norm(
                                    x,
                                    is_training=True,
                                    momentum=0.99,
                                    epsilon=1e-5,
                                    name="batch_norm_RB_3"+'-'+str(order)
                                    )
        logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    identity = mtf.reshape(
                            identity, 
                            new_shape=[identity.shape.dims[0],identity.shape.dims[1],identity.shape.dims[2], x.shape.dims[3]],
                            name="reshape_RB"+str(order)
                            )
    logger.debug("[output tensor] (name,shape):({},{})".format(identity.name,identity.shape))
    x = mtf.add(x,identity,output_shape=x.shape,name="add_RB_1"+'-'+str(order))
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_RB_3"+'-'+str(order))
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    return x


def BasicBlock(x, order, out_channels, strides,float16=None,batch_norm=False):
    name = "BasicBlock"
    expansion = 1
    out_chls = out_channels // expansion
    identity = x

    x = conv2d(
                            x, 
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters1',
                                                        size=out_chls
                                                        ),
                            filter_size=(3,3),
                            strides=strides,
                            name="conv3x3_BB_1"+'-'+str(order),
                            variable_dtype=float16
                            )
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    if batch_norm:
        x,_ = mtf.layers.batch_norm(
                                    x,
                                    is_training=True,
                                    momentum=0.99,
                                    epsilon=1e-5,
                                    name="batch_norm_BB_1"+'-'+str(order)
                                    )
        logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_BB_1"+'-'+str(order))
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters2',
                                                        size=out_channels
                                                        ),
                            filter_size=(3,3),
                            strides=(1,1),
                            name="conv3x3_BB_2"+'-'+str(order),
                            variable_dtype=float16
                            )
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    if batch_norm:
        x,_ = mtf.layers.batch_norm(
                                    x,
                                    is_training=True,
                                    momentum=0.99,
                                    epsilon=1e-5,
                                    name="batch_norm_BB_2"+'-'+str(order)
                                    )
        logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    identity = mtf.reshape(
                            identity, 
                            new_shape=[identity.shape.dims[0],identity.shape.dims[1],identity.shape.dims[2], x.shape.dims[3]],
                            name="reshape_BB"+str(order)
                            )
    logger.debug("[output tensor] (name,shape):({},{})".format(identity.name,identity.shape))
    x = mtf.add(x,identity,output_shape=x.shape,name="add_BB_1"+'-'+str(order))
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_BB_2"+'-'+str(order))
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    
    return x



def BasicBlockWithDown(x, order, out_channels, strides,float16=None,batch_norm=False):
    name = "BasicBlockWithDown"
    expansion = 1
    out_chls = out_channels // expansion
    identity = x

    x = conv2d(
                            x, 
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters1',
                                                        size=out_chls
                                                        ),
                            filter_size=(3,3),
                            strides=(1,1),
                            name="conv3x3_BBW_1"+'-'+str(order),
                            variable_dtype=float16
                            )
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    if batch_norm:
        x,_ = mtf.layers.batch_norm(
                                    x,
                                    is_training=True,
                                    momentum=0.99,
                                    epsilon=1e-5,
                                    name="batch_norm_BBW_1"+'-'+str(order)
                                    )
        logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_BBW_1"+'-'+str(order))
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    

    x = conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters2',
                                                        size=out_channels
                                                        ),
                            filter_size=(3,3),
                            strides=strides,
                            name="conv3x3_BBW_2"+'-'+str(order),
                            variable_dtype=float16
                            )
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    if batch_norm:
        x,_ = mtf.layers.batch_norm(
                                    x,
                                    is_training=True,
                                    momentum=0.99,
                                    epsilon=1e-5,
                                    name="batch_norm_BBW_2"+'-'+str(order)
                                    )
        logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    identity = conv2d(
                                    identity,
                                    output_dim=mtf.Dimension(
                                                                name=name+'-'+str(order)+'-'+'filters3',
                                                                size=out_channels
                                                                ),
                                    filter_size=(1,1),
                                    strides=strides,
                                    name="conv1x1_BBW_1"+'-'+str(order),
                                    variable_dtype=float16
                                    )
    logger.debug("[output tensor] (name,shape):({},{})".format(identity.name,identity.shape))
    if batch_norm:
        identity,_ = mtf.layers.batch_norm(
                                            identity,
                                            is_training=True,
                                            momentum=0.99,
                                            epsilon=1e-5,
                                            name="batch_norm_BBW_3"+'-'+str(order)
                                            )
        logger.debug("[output tensor] (name,shape):({},{})".format(identity.name,identity.shape))
    identity = mtf.reshape(
                            identity, 
                            new_shape=[identity.shape.dims[0],identity.shape.dims[1],identity.shape.dims[2], x.shape.dims[3]],
                            name="reshape_BBW"+str(order)
                            )
    logger.debug("[output tensor] (name,shape):({},{})".format(identity.name,identity.shape))
    x = mtf.add(x,identity,output_shape=x.shape,name="add_BBW_1"+'-'+str(order))
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_BBW_2"+'-'+str(order))
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    return x


def backbone(x, layerlist, chalist, strilist, classes_dim, blocklist, float16=None,batch_norm=False):
    name = "backbone"
    
    x = conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+'filters',
                                                        size=64
                                                        ),
                            filter_size=(7,7),
                            strides=(2,2),
                            # padding="VALID",
                            name="conv7x7_backbone",
                            variable_dtype=float16
                            )
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    if batch_norm:
        x,_ = mtf.layers.batch_norm(
                                    x,
                                    is_training=True,
                                    momentum=0.99,
                                    epsilon=1e-5,
                                    name="batch_norm_backbone"
                                    )
        logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_backbone")
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.layers.max_pool2d(
                                x,
                                ksize=(2,2),
                                name="maxpool_backbone"
                                )
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    shortcuttype1 = 0
    shortcuttype2 = 0
    for _,(layer, channel, strides) in enumerate(zip(layerlist, chalist, strilist)):
        x = blocklist[0](x,order=shortcuttype1,out_channels=channel,strides=(strides,strides),float16=float16,batch_norm=batch_norm)
        shortcuttype1+=1
        for _ in range(layer-1):
            x = blocklist[1](x, order= shortcuttype2,out_channels=channel,strides=(1,1),float16=float16,batch_norm=batch_norm)
            shortcuttype2+=1
    
    # x = mtf.einsum([x], output_shape=[list(x.shape.dims)[0],list(x.shape.dims)[3]], name="einsum_backbone")
    x = mtf.layers.avg_pool2d(x,ksize=(x.shape.dims[1].size,x.shape.dims[2].size))
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
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
    logit = mtf.layers.dense(x, classes_dim, name="dense_backbone",variable_dtype=float16)
    logger.debug("[output tensor] (name,shape):({},{})".format(logit.name,logit.shape))
    return logit


def resnet18(x, classes_dim, float16=None, batch_norm=False):
    if float16:
        x = mtf.cast(x,dtype=tf.float16)
    logger.debug("[input tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = backbone(
                    x,
                    layerlist=[2,2,2,2],
                    chalist=[64,128,256,512],
                    strilist=[1,2,2,2],
                    classes_dim=classes_dim,
                    blocklist=[BasicBlockWithDown,BasicBlock],
                    float16=float16,
                    batch_norm=batch_norm
                    )
    return x
def resnet34(x, classes_dim,float16=None, batch_norm=False):
    if float16:
        x = mtf.cast(x,dtype=tf.float16)
    logger.debug("[input tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = backbone(
                    x,
                    layerlist=[3,4,6,3],
                    chalist=[64,128,256,512],
                    strilist=[1,2,2,2],
                    classes_dim=classes_dim,
                    blocklist=[BasicBlockWithDown,BasicBlock],
                    float16=float16,
                    batch_norm=batch_norm
                    )
    return x
def resnet50(x, classes_dim,float16=None, batch_norm=False):
    if float16:
        x = mtf.cast(x,dtype=tf.float16)
    logger.debug("[input tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = backbone(
                    x,
                    layerlist=[3,4,6,3],
                    chalist=[256,512,1024,2048],
                    strilist=[1,2,2,2],
                    classes_dim=classes_dim,
                    blocklist=[ResidualBlockWithDown,ResidualBlock],
                    float16=float16,
                    batch_norm=batch_norm
                    )
    return x
def resnet101(x, classes_dim,float16=None, batch_norm=False):
    if float16:
        x = mtf.cast(x,dtype=tf.float16)
    logger.debug("[input tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = backbone(
                    x,
                    layerlist=[3,4,23,3],
                    chalist=[256,512,1024,2048],
                    strilist=[1,2,2,2],
                    classes_dim=classes_dim,
                    blocklist=[ResidualBlockWithDown,ResidualBlock],
                    float16=float16,
                    batch_norm=batch_norm
                    )
    return x
def resnet152(x, classes_dim,float16=None, batch_norm=False):
    if float16:
        x = mtf.cast(x,dtype=tf.float16)
    logger.debug("[input tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = backbone(
                    x,
                    layerlist=[3,8,36,3],
                    chalist=[256,512,1024,2048],
                    strilist=[1,2,2,2],
                    classes_dim=classes_dim,
                    blocklist=[ResidualBlockWithDown,ResidualBlock],
                    float16=float16,
                    batch_norm=batch_norm
                    )
    return x

