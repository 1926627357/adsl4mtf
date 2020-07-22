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
                            strides=strides,
                            name="conv1x1"
                            )
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm"
                                )
    x = mtf.relu(x)

    x = mtf.layers.conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters2',
                                                        size=out_chls
                                                        ),
                            filter_size=(3,3),
                            strides=(1,1),
                            name="conv3x3"
                            )
    x = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm"
                                )
    x = mtf.relu(x)

    x = mtf.layers.conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters3',
                                                        size=out_channels
                                                        ),
                            filter_size=(1,1),
                            strides=(1,1),
                            name="conv1x1"
                            )
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm"
                                )
    identity = mtf.layers.conv2d(
                                    identity,
                                    output_dim=mtf.Dimension(
                                                                name=name+'-'+str(order)+'-'+'filters4',
                                                                size=out_channels
                                                                ),
                                    filter_size=(1,1),
                                    strides=strides,
                                    name="conv1x1"
                                    )
    identity,_ = mtf.layers.batch_norm(
                                        identity,
                                        is_training=True,
                                        momentum=0.99,
                                        epsilon=1e-5,
                                        name="batch_norm"
                                        )

    x = x + identity
    x = mtf.relu(x)
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
                            strides=strides,
                            name="conv1x1"
                            )
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm"
                                )
    x = mtf.relu(x)

    x = mtf.layers.conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters2',
                                                        size=out_chls
                                                        ),
                            filter_size=(3,3),
                            strides=(1,1),
                            name="conv3x3"
                            )
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm"
                                )
    x = mtf.relu(x)

    x = mtf.layers.conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+str(order)+'-'+'filters3',
                                                        size=out_channels
                                                        ),
                            filter_size=(1,1),
                            strides=(1,1),
                            name="conv1x1"
                            )
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm"
                                )
    x = x + identity
    x = mtf.relu(x)
    return x

def backbone(x, layerlist, chalist, strilist, classes_dim):
    name = "backbone"
    x = mtf.layers.conv2d(
                            x,
                            output_dim=mtf.Dimension(
                                                        name=name+'-'+'filters',
                                                        size=64
                                                        ),
                            filter_size=(7,7),
                            strides=(2,2),
                            name="conv7x7"
                            )
    x,_ = mtf.layers.batch_norm(
                                x,
                                is_training=True,
                                momentum=0.99,
                                epsilon=1e-5,
                                name="batch_norm"
                                )
    x = mtf.relu(x,name="relu")
    x = mtf.layers.max_pool2d(
                                x,
                                ksize=(3,3),
                                name="maxpool"
                                )
    ResidualBlockWithDown_order = 0
    ResidualBlock_order = 0
    
    for layer, channel, strides in zip(layerlist, chalist, strilist):
        x = ResidualBlockWithDown(x,order=ResidualBlockWithDown_order,out_channels=channel,strides=(strides,strides))
        ResidualBlockWithDown_order +=1
        for _ in range(layer-1):
            x = ResidualBlock(x, order=ResidualBlock_order,out_channels=channel,strides=(1,1))
            ResidualBlock_order += 1
    x = mtf.einsum(x, output_shape=mtf.Shape([x.shape.dims[0],x.shape.dims[1]]), name="einsum")

    logit = mtf.layers.dense(x, classes_dim, name="dense")
    return logit

def resnet_model(x, classes_dim, depth):
    if depth not in white_list:
        print("Renet-{}".format(depth))
        raise ValueError
    else:
        if depth==50:
        # resnet50
            x = backbone(
                            x,
                            layerlist=[3,4,6,3],
                            chalist=[256,512,1024,2048],
                            strilist=[1,2,2,2],
                            classes_dim=classes_dim
                            )
    return x