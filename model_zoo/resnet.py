import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf
import adsl4mtf.log as logger
white_list = [18,34,50,101,152]



def ResidualBlockWithDown(x, order, out_channels, strides):
    name = "ResidualBlockWithDown"
    expansion = 4
    out_chls = out_channels // expansion
    identity = x

    x = mtf.layers.conv2d(
                            x, 
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters1',
                                                        size=out_chls
                                                        ),
                            filter_size=(1,1),
                            strides=(1,1),
                            name="conv1x1_RBW_1"+'-'+str(order)
                            )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_RBW_1"+'-'+str(order)
                                )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_RBW_1"+'-'+str(order))
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.layers.conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters2',
                                                        size=out_chls
                                                        ),
                            filter_size=(3,3),
                            strides=strides,
                            name="conv3x3_RBW_1"+'-'+str(order)
                            )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_RBW_2"+'-'+str(order)
                                )

    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_RBW_2"+'-'+str(order))
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.layers.conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters3',
                                                        size=out_channels
                                                        ),
                            filter_size=(1,1),
                            strides=(1,1),
                            name="conv1x1-2_RBW_2"+'-'+str(order)
                            )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_RBW_3"+'-'+str(order)
                                )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    identity = mtf.layers.conv2d(
                                    identity,
                                    output_dim=mtf.Dimension(
                                                                name=name+'-'+str(order)+'-'+'filters3',
                                                                size=out_channels
                                                                ),
                                    filter_size=(1,1),
                                    strides=strides,
                                    name="conv1x1_RBW_3"+'-'+str(order)
                                    )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    identity,_ = mtf.layers.batch_norm(
                                        identity,
                                        is_training=True,
                                        momentum=0.99,
                                        epsilon=1e-5,
                                        name="batch_norm_RBW_4"+'-'+str(order)
                                        )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    identity = mtf.reshape(
                            identity, 
                            new_shape=[identity.shape.dims[0],identity.shape.dims[1],identity.shape.dims[2], x.shape.dims[3]],
                            name="reshape_RBW"+str(order)
                            )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.add(x,identity,output_shape=x.shape,name="add_RBW_1"+'-'+str(order))
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_RBW_3"+'-'+str(order))
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    return x

def ResidualBlock(x, order, out_channels, strides):
    name = "ResidualBlock"
    expansion = 4
    out_chls = out_channels // expansion
    identity = x

    x = mtf.layers.conv2d(
                            x, 
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters1',
                                                        size=out_chls
                                                        ),
                            filter_size=(1,1),
                            strides=(1,1),
                            name="conv1x1_RB_1"+'-'+str(order)
                            )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_RB_1"+'-'+str(order)
                                )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_RB_1"+'-'+str(order))
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.layers.conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters2',
                                                        size=out_chls
                                                        ),
                            filter_size=(3,3),
                            strides=strides,
                            name="conv3x3_RB_1"+'-'+str(order)
                            )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_RB_2"+'-'+str(order)
                                )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_RB_2"+'-'+str(order))
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.layers.conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters3',
                                                        size=out_channels
                                                        ),
                            filter_size=(1,1),
                            strides=(1,1),
                            name="conv1x1_RB_2"+'-'+str(order)
                            )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_RB_3"+'-'+str(order)
                                )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    identity = mtf.reshape(
                            identity, 
                            new_shape=[identity.shape.dims[0],identity.shape.dims[1],identity.shape.dims[2], x.shape.dims[3]],
                            name="reshape_RB"+str(order)
                            )
    logger.info("[output tensor] (name,shape):({},{})".format(identity.name,identity.shape))
    x = mtf.add(x,identity,output_shape=x.shape,name="add_RB_1"+'-'+str(order))
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_RB_3"+'-'+str(order))
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    return x


def BasicBlock(x, order, out_channels, strides):
    name = "BasicBlock"
    expansion = 1
    out_chls = out_channels // expansion
    identity = x

    x = mtf.layers.conv2d(
                            x, 
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters1',
                                                        size=out_chls
                                                        ),
                            filter_size=(3,3),
                            strides=strides,
                            name="conv3x3_BB_1"+'-'+str(order)
                            )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_BB_1"+'-'+str(order)
                                )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_BB_1"+'-'+str(order))
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.layers.conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters2',
                                                        size=out_channels
                                                        ),
                            filter_size=(3,3),
                            strides=(1,1),
                            name="conv3x3_BB_2"+'-'+str(order)
                            )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_BB_2"+'-'+str(order)
                                )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    identity = mtf.reshape(
                            identity, 
                            new_shape=[identity.shape.dims[0],identity.shape.dims[1],identity.shape.dims[2], x.shape.dims[3]],
                            name="reshape_BB"+str(order)
                            )
    logger.info("[output tensor] (name,shape):({},{})".format(identity.name,identity.shape))
    x = mtf.add(x,identity,output_shape=x.shape,name="add_BB_1"+'-'+str(order))
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_BB_2"+'-'+str(order))
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    
    return x



def BasicBlockWithDown(x, order, out_channels, strides):
    name = "BasicBlockWithDown"
    expansion = 1
    out_chls = out_channels // expansion
    identity = x

    x = mtf.layers.conv2d(
                            x, 
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters1',
                                                        size=out_chls
                                                        ),
                            filter_size=(3,3),
                            strides=(1,1),
                            name="conv3x3_BBW_1"+'-'+str(order)
                            )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_BBW_1"+'-'+str(order)
                                )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_BBW_1"+'-'+str(order))
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    

    x = mtf.layers.conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters2',
                                                        size=out_channels
                                                        ),
                            filter_size=(3,3),
                            strides=strides,
                            name="conv3x3_BBW_2"+'-'+str(order)
                            )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_BBW_2"+'-'+str(order)
                                )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    identity = mtf.layers.conv2d(
                                    identity,
                                    output_dim=mtf.Dimension(
                                                                name=name+'-'+str(order)+'-'+'filters3',
                                                                size=out_channels
                                                                ),
                                    filter_size=(1,1),
                                    strides=strides,
                                    name="conv1x1_BBW_1"+'-'+str(order)
                                    )
    logger.info("[output tensor] (name,shape):({},{})".format(identity.name,identity.shape))
    identity,_ = mtf.layers.batch_norm(
                                        identity,
                                        is_training=True,
                                        momentum=0.99,
                                        epsilon=1e-5,
                                        name="batch_norm_BBW_3"+'-'+str(order)
                                        )
    logger.info("[output tensor] (name,shape):({},{})".format(identity.name,identity.shape))
    identity = mtf.reshape(
                            identity, 
                            new_shape=[identity.shape.dims[0],identity.shape.dims[1],identity.shape.dims[2], x.shape.dims[3]],
                            name="reshape_BBW"+str(order)
                            )
    logger.info("[output tensor] (name,shape):({},{})".format(identity.name,identity.shape))
    x = mtf.add(x,identity,output_shape=x.shape,name="add_BBW_1"+'-'+str(order))
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_BBW_2"+'-'+str(order))
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    return x


def backbone(x, layerlist, chalist, strilist, classes_dim, blocklist):
    name = "backbone"
    
    x = mtf.layers.conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+'filters',
                                                        size=64
                                                        ),
                            filter_size=(7,7),
                            strides=(2,2),
                            # padding="VALID",
                            name="conv7x7_backbone"
                            )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_backbone"
                                )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.relu(x,name="relu_backbone")
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.layers.max_pool2d(
                                x,
                                ksize=(2,2),
                                name="maxpool_backbone"
                                )
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    shortcuttype1 = 0
    shortcuttype2 = 0
    for index,(layer, channel, strides) in enumerate(zip(layerlist, chalist, strilist)):
        x = blocklist[0](x,order=shortcuttype1,out_channels=channel,strides=(strides,strides))
        shortcuttype1+=1
        for tindex in range(layer-1):
            x = blocklist[1](x, order= shortcuttype2,out_channels=channel,strides=(1,1))
            shortcuttype2+=1
    
    # x = mtf.einsum([x], output_shape=[list(x.shape.dims)[0],list(x.shape.dims)[3]], name="einsum_backbone")
    x = mtf.layers.avg_pool2d(x,ksize=(x.shape.dims[1].size,x.shape.dims[2].size))
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
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
    logger.info("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    logit = mtf.layers.dense(x, classes_dim, name="dense_backbone")
    logger.info("[output tensor] (name,shape):({},{})".format(logit.name,logit.shape))
    return logit

def resnet_model(x, classes_dim, depth):
    logger.info("[input tensor] (name,shape):({},{})".format(x.name,x.shape))
    if depth not in white_list:
        logger.error("Renet{} are not supported".format(depth))
        raise ValueError
    else:
        if depth==18:
        # resnet18
            x = backbone(
                            x,
                            layerlist=[2,2,2,2],
                            chalist=[64,128,256,512],
                            strilist=[1,2,2,2],
                            classes_dim=classes_dim,
                            blocklist=[BasicBlockWithDown,BasicBlock]
                            )
        if depth==34:
        # resnet34
            x = backbone(
                            x,
                            layerlist=[3,4,6,3],
                            chalist=[64,128,256,512],
                            strilist=[1,2,2,2],
                            classes_dim=classes_dim,
                            blocklist=[BasicBlockWithDown,BasicBlock]
                            )



        if depth==50:
        # resnet50
            x = backbone(
                            x,
                            layerlist=[3,4,6,3],
                            chalist=[256,512,1024,2048],
                            strilist=[1,2,2,2],
                            classes_dim=classes_dim,
                            blocklist=[ResidualBlockWithDown,ResidualBlock]
                            )

        if depth==101:
        # resnet101
            x = backbone(
                            x,
                            layerlist=[3,4,23,3],
                            chalist=[256,512,1024,2048],
                            strilist=[1,2,2,2],
                            classes_dim=classes_dim,
                            blocklist=[ResidualBlockWithDown,ResidualBlock]
                            )
        if depth==152:
        # resnet152
            x = backbone(
                            x,
                            layerlist=[3,8,36,3],
                            chalist=[256,512,1024,2048],
                            strilist=[1,2,2,2],
                            classes_dim=classes_dim,
                            blocklist=[ResidualBlockWithDown,ResidualBlock]
                            )
    return x