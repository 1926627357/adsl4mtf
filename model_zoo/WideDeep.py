import mesh_tensorflow as mtf
import sys
import os
sys.path.append(os.path.abspath('.'))
import tensorflow.compat.v1 as tf
import numpy as np
import adsl4mtf.log as logger
def deep(x, mask, float16=None):
    x = mtf.einsum([x,mask],output_shape=x.shape.dims, name='deep_mul')
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))

    # 使用仿照mindspore中使用fp16来计算下面的dense
    x = mtf.cast(x,dtype=tf.float16)
    
    x = mtf.layers.dense(x, mtf.Dimension(
                                            name='dense_dim0',size=1024),
                                            name="deep-dense-0",
                                            reduced_dims=x.shape.dims[-2:],
                                            activation=mtf.relu,
                                            variable_dtype=mtf.VariableDType(tf.float16,tf.float16,tf.float16))
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.layers.dense(x, mtf.Dimension(
                                            name='dense_dim1',size=512),
                                            name="deep-dense-1",
                                            activation=mtf.relu,
                                            variable_dtype=mtf.VariableDType(tf.float16,tf.float16,tf.float16))
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.layers.dense(x, mtf.Dimension(
                                            name='dense_dim2',size=256),
                                            name="deep-dense-2",
                                            activation=mtf.relu,
                                            variable_dtype=mtf.VariableDType(tf.float16,tf.float16,tf.float16))
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.layers.dense(x, mtf.Dimension(
                                            name='dense_dim3',size=128),
                                            name="deep-dense-3",
                                            activation=mtf.relu,
                                            variable_dtype=mtf.VariableDType(tf.float16,tf.float16,tf.float16))
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    x = mtf.layers.dense(x, mtf.Dimension(
                                            name='dense_dim4',size=1),
                                            name="deep-dense-4",
                                            variable_dtype=mtf.VariableDType(tf.float16,tf.float16,tf.float16))
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    if float16:
        pass
    else:
        x = mtf.cast(x,dtype=tf.float32)
    return x
def wide(x, mask, float16=None):
    x = mtf.einsum([x,mask],output_shape=[x.shape.dims[0],x.shape.dims[-1]], name='wide_mul')
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    if float16:
        wide_b = np.array(0,dtype=np.float16)
    else:
        wide_b = np.array(0,dtype=np.float32)

    x = mtf.add(x,wide_b,name="wide_sum")
    logger.debug("[output tensor] (name,shape):({},{})".format(x.name,x.shape))
    return x


def widedeep(id_hldr, wt_hldr, vocab_dim, embed_dim, outdim, float16=None):
    logger.debug("[input tensor] (name,shape):({},{})".format(id_hldr.name,id_hldr.shape))
    logger.debug("[input tensor] (name,shape):({},{})".format(wt_hldr.name,wt_hldr.shape))
    if float16:
        embedding_table = mtf.layers.embedding_weights(id_hldr.mesh, vocab_dim, embed_dim, float16, "deep_embedding")
        deep_output = mtf.gather(embedding_table, id_hldr, vocab_dim)
        # deep_output = mtf.layers.embedding(id_hldr, vocab_dim=vocab_dim, output_dim=embed_dim, variable_dtype=float16, name="deep_embedding")
    else:
        fp32 = mtf.VariableDType(tf.float32,tf.float32,tf.float32)
        embedding_table = mtf.layers.embedding_weights(id_hldr.mesh, vocab_dim, embed_dim, fp32, "deep_embedding")
        deep_output = mtf.gather(embedding_table, id_hldr, vocab_dim)
        # deep_output = mtf.layers.embedding(id_hldr, vocab_dim=vocab_dim, output_dim=embed_dim, variable_dtype=fp32, name="deep_embedding")
    logger.debug("[output tensor] (name,shape):({},{})".format(deep_output.name,deep_output.shape))
    expend_dim = mtf.Dimension('expend',size=1)
    embed_dim_one = mtf.Dimension('embed_dim_one',size=1)
    mask = mtf.reshape(wt_hldr, new_shape=[wt_hldr.shape.dims[0],wt_hldr.shape.dims[1],expend_dim], name='mask_reshape')
    logger.debug("[output tensor] (name,shape):({},{})".format(mask.name,mask.shape))
    if float16:
        wide_output = mtf.layers.embedding(id_hldr, vocab_dim=vocab_dim, output_dim=embed_dim_one, variable_dtype=float16, name="wide_embedding")
    else:
        fp32 = mtf.VariableDType(tf.float32,tf.float32,tf.float32)
        wide_output = mtf.layers.embedding(id_hldr, vocab_dim=vocab_dim, output_dim=embed_dim_one, variable_dtype=fp32, name="wide_embedding")
    logger.debug("[output tensor] (name,shape):({},{})".format(wide_output.name,wide_output.shape))

    wide_output = wide(wide_output,mask=mask,float16=float16)
    deep_output = deep(deep_output,mask=mask,float16=float16)
    
    result = mtf.add(wide_output,deep_output)
    result = mtf.reshape(result, new_shape=[wide_output.shape.dims[0],outdim],name='result_reshape')
    result = mtf.reduce_sum(result,reduced_dim=outdim)
    logger.debug("[output tensor] (name,shape):({},{})".format(result.name, result.shape))
    return result,embedding_table


if __name__=="__main__":
    
    
    os.environ['GLOG_v'] = '0'
    # global_step = tf.train.get_global_step()
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    batch_dim = mtf.Dimension("batch",10)
    field_dim = mtf.Dimension("field",39)
    vocab_dim = mtf.Dimension("vocab_size",20000)
    embed_dim = mtf.Dimension("embed_size",80)
    outdim = mtf.Dimension("outdim",1)
    # label_dim = mtf.Dimension("label",1)

    # input1 = mtf.get_variable(mesh, "input1", [dim1,dim2])
    # input2 = mtf.get_variable(mesh, "input2", [dim3,dim4])

    id_hldr = mtf.import_tf_tensor(mesh, np.random.randn(batch_dim.size,field_dim.size).astype(np.int32), shape=[batch_dim,field_dim])
    wt_hldr = mtf.import_tf_tensor(mesh, np.random.randn(batch_dim.size,field_dim.size).astype(np.float32), shape=[batch_dim,field_dim])
    label = mtf.import_tf_tensor(mesh, np.array([1,0,1,0,1,0,1,0,1,0]).astype(np.float32), shape=[batch_dim])

    fp16=True
    if fp16:
        float16=mtf.VariableDType(tf.float16,tf.float16,tf.float16)
        id_hldr=mtf.cast(id_hldr,dtype=tf.int32)
        wt_hldr=mtf.cast(wt_hldr,dtype=tf.float16)
    else:
        float16=None
    result,embedding_table = widedeep(id_hldr,wt_hldr,vocab_dim,embed_dim,outdim,float16)
    
    # label = mtf.reshape(label,new_shape=[batch_dim, outdim])
    # output = mtf.layers.softmax_cross_entropy_with_logits(result, label,vocab_dim=outdim)
    # result = mtf.sigmoid(result)
    # result = -(label*mtf.log(result)+(1-label)*mtf.log(1-result))
    # result = mtf.reduce_sum(result)
    result = mtf.cast(result,dtype=tf.float32)
    embedding_table = mtf.cast(embedding_table,dtype=tf.float32)
    probability = mtf.sigmoid(result)
    result = mtf.layers.sigmoid_cross_entropy_with_logits(result,label)
    wide_loss = mtf.reduce_mean(result)
    deep_loss = mtf.reduce_mean(mtf.square(embedding_table)) / 2
    deep_loss = mtf.reduce_mean(result)+8e-5*deep_loss

    

    # print("========",global_step)
    devices = ["gpu:0"]
    mesh_shape = [("all_processors", 1)]
    layout_rules = [("dim1", "all_processors")]
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        mesh_shape, layout_rules, devices)
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    wide_loss = lowering.export_to_tf_tensor(wide_loss)
    # result = lowering.export_to_tf_tensor(result)
    # predict = lowering.export_to_tf_tensor(probability)
    # predict = tf.where(predict>0.5,tf.ones_like(predict),tf.zeros_like(predict))
    deep_loss = lowering.export_to_tf_tensor(deep_loss)
    print(wide_loss)
    print(deep_loss)
    # print(result)
    # print(predict)
