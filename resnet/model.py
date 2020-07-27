import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf
white_list = [50,101,152]



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
    print(x.name)
    print(x.shape)
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_RBW_1"+'-'+str(order)
                                )
    
    x = mtf.relu(x,name="relu_RBW_1"+'-'+str(order))
    
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
    print(x.name)
    print(x.shape)
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_RBW_2"+'-'+str(order)
                                )
    x = mtf.relu(x,name="relu_RBW_2"+'-'+str(order))

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
    print(x.name)
    print(x.shape)
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_RBW_3"+'-'+str(order)
                                )
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
    print(identity.name)
    print(identity.shape)
    identity,_ = mtf.layers.batch_norm(
                                        identity,
                                        is_training=True,
                                        momentum=0.99,
                                        epsilon=1e-5,
                                        name="batch_norm_RBW_4"+'-'+str(order)
                                        )

    x = mtf.add(x,identity,output_shape=x.shape,name="add_RBW_1"+'-'+str(order))
    x = mtf.relu(x,name="relu_RBW_3"+'-'+str(order))
    print(x.name)
    print(x.shape)
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
    print(x.name)
    print(x.shape)
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_RB_1"+'-'+str(order)
                                )
    x = mtf.relu(x,name="relu_RB_1"+'-'+str(order))

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
    print(x.name)
    print(x.shape)
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_RB_2"+'-'+str(order)
                                )
    x = mtf.relu(x,name="relu_RB_2"+'-'+str(order))

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
    print(x.name)
    print(x.shape)
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_RB_3"+'-'+str(order)
                                )
    identity = mtf.reshape(
                            identity, 
                            new_shape=[identity.shape.dims[0],identity.shape.dims[1],identity.shape.dims[2], x.shape.dims[3]],
                            name="reshape_RB"+str(order)
                            )
    x = mtf.add(x,identity,output_shape=x.shape,name="add_RB_1"+'-'+str(order))
    x = mtf.relu(x,name="relu_RB_3"+'-'+str(order))
    print(x.name)
    print(x.shape)
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
    print(x.name)
    print(x.shape)
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_BB_1"+'-'+str(order)
                                )
    x = mtf.relu(x,name="relu_BB_1"+'-'+str(order))

    x = mtf.layers.conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters3',
                                                        size=out_channels
                                                        ),
                            filter_size=(3,3),
                            strides=(1,1),
                            name="conv1x1_BB_2"+'-'+str(order)
                            )
    print(x.name)
    print(x.shape)
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_BB_2"+'-'+str(order)
                                )
    identity = mtf.reshape(
                            identity, 
                            new_shape=[identity.shape.dims[0],identity.shape.dims[1],identity.shape.dims[2], x.shape.dims[3]],
                            name="reshape_BB"+str(order)
                            )
    x = mtf.add(x,identity,output_shape=x.shape,name="add_BB_1"+'-'+str(order))
    x = mtf.relu(x,name="relu_BB_2"+'-'+str(order))
    print(x.name)
    print(x.shape)
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
    print(x.name)
    print(x.shape)
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_BBW_1"+'-'+str(order)
                                )
    
    x = mtf.relu(x,name="relu_BBW_1"+'-'+str(order))
    
    

    x = mtf.layers.conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters2',
                                                        size=out_channels
                                                        ),
                            filter_size=(3,3),
                            strides=strides,
                            name="conv3x3-2_BBW_2"+'-'+str(order)
                            )
    print(x.name)
    print(x.shape)
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_BBW_2"+'-'+str(order)
                                )
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
    print(identity.name)
    print(identity.shape)
    identity,_ = mtf.layers.batch_norm(
                                        identity,
                                        is_training=True,
                                        momentum=0.99,
                                        epsilon=1e-5,
                                        name="batch_norm_BBW_3"+'-'+str(order)
                                        )

    x = mtf.add(x,identity,output_shape=x.shape,name="add_BBW_1"+'-'+str(order))
    x = mtf.relu(x,name="relu_BBW_2"+'-'+str(order))
    print(x.name)
    print(x.shape)
    return x


def backbone(x, layerlist, chalist, strilist, classes_dim):
    name = "backbone"
    print(x.name)
    print(x.shape)
    x = mtf.layers.conv2d_with_blocks(
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
    print(x.name)
    print(x.shape)
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm_backbone"
                                )
    x = mtf.relu(x,name="relu_backbone")
    print(x.name)
    print(x.shape)
    x = mtf.layers.max_pool2d(
                                x,
                                ksize=(2,2),
                                name="maxpool_backbone"
                                )
    print(x.name)
    print(x.shape)
    
    for index,(layer, channel, strides) in enumerate(zip(layerlist, chalist, strilist)):
        x = ResidualBlockWithDown(x,order=index,out_channels=channel,strides=(strides,strides))
        for tindex in range(layer-1):
            x = ResidualBlock(x, order= index * layer +tindex+1,out_channels=channel,strides=(1,1))
    
    x = mtf.einsum([x], output_shape=[list(x.shape.dims)[0],list(x.shape.dims)[3]], name="einsum_backbone")

    logit = mtf.layers.dense(x, classes_dim, name="dense_backbone")
    print(logit.name)
    print(logit.shape)
    return logit

def resnet_model(x, classes_dim, depth):
    print(x.name)
    print(x.shape)
    if depth not in white_list:
        print("Renet-{}".format(depth))
        raise ValueError
    else:
        if depth==50:
        # resnet50
            x = backbone(
                            x,
                            layerlist=[2,2,2,2],
                            chalist=[256,512,1024,2048],
                            strilist=[1,2,2,2],
                            classes_dim=classes_dim
                            )
    return x