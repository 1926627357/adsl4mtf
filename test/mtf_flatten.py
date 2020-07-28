import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()
# batch_size = 64
batch_size = mtf.Dimension("batch_size",64)
# image_height = 1
image_height = mtf.Dimension("image_height",1)
# image_width = 1
image_width = mtf.Dimension("image_width",1)
# channels = 1
channels = mtf.Dimension("channels",3)
kh = 1
kw = 1
filters = mtf.Dimension("filters",3)


graph = mtf.Graph()
mesh = mtf.Mesh(graph, "my_mesh")
input = mtf.get_variable(mesh, "input", [batch_size, image_height, image_width, channels],dtype=tf.float16)

# output = mtf.layers.conv2d(
#                             input, 
#                             output_dim=filters,
#                             filter_size=(kh,kw),
#                             strides=(2,2),
#                             name="conv2d-{}x{}".format(kh,kw)
#                             )

output = mtf.ops._tf_flatten_batch_dims(input, num_nonbatch_dims=1)
print("[intra variable]:",output)

variables = graph._all_variables
print("[variable num]: {}".format(len(variables)))
for v in variables:
	print("[variable] : ({})".format(v))




