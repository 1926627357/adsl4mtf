from mesh_tensorflow import ops_with_redefined_builtins as mtf

import tensorflow.compat.v1 as tf
import itertools


import math

from mesh_tensorflow import Operation, Dimension, conv2d_backprop_input, conv2d_backprop_filter, Shape, Tensor, LazyAllreduceSum



class Conv2dOperation(Operation):
  """like tf.nn.conv2d.

  Always data format "NHWC".
  # TODO(nikip): support dilations
  Always dilation rate of 1
  padding: "SAME" or "VALID"

  TODO(noam): implement more options.
  """

  def __init__(self, conv_input, conv_filter, strides, padding, name=None):
    super(Conv2dOperation, self).__init__(
        [conv_input, conv_filter], name=name or "conv2d")
    self._padding = padding
    self._batch_dims = conv_input.shape.dims[:-3]
    self._in_h_dim, self._in_w_dim, self._in_dim = conv_input.shape.dims[-3:]
    self._fh_dim, self._fw_dim = conv_filter.shape.dims[:2]
    f_in_dim, self._out_dim = conv_filter.shape.dims[2:]
    if f_in_dim != self._in_dim:
      raise ValueError("Dimensions do not match input=%s filter=%s"
                       % (conv_input, conv_filter))
    out_h = self._in_h_dim.size
    out_w = self._in_w_dim.size
    if padding == "VALID":
      out_h -= (self._fh_dim.size - 1)
      out_w -= (self._fw_dim.size - 1)

    self._strides = strides
    if strides is not None:
      
      # out_h = out_h // strides[1] if out_h>1 else out_h
      out_h = math.ceil(out_h/strides[1])
      # print("out_h: ",out_h)
      # out_w = out_w // strides[2] if out_w>1 else out_w
      out_w = math.ceil(out_w/strides[2])
      # print("out_w: ",out_w)
    self._out_h_dim = Dimension(self._in_h_dim.name, out_h)
    self._out_w_dim = Dimension(self._in_w_dim.name, out_w)
    output_shape = Shape(
        self._batch_dims + [self._out_h_dim, self._out_w_dim, self._out_dim])
    self._outputs = [Tensor(self, output_shape, conv_input.dtype)]

    unsplittable_dims = [self._in_h_dim, self._in_w_dim, self._fh_dim,
                         self._fw_dim]
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_splittable_and_unsplittable_dims(
            "splittable", [dim.name for dim in unsplittable_dims]))

  def gradient(self, grad_ys):
    dy = grad_ys[0]
    conv_input, conv_filter = self.inputs
    return [
        conv2d_backprop_input(self._inputs[0].shape,
                              conv_filter,
                              dy,
                              self._strides,
                              self._padding),
        conv2d_backprop_filter(conv_input,
                               self._inputs[1].shape,
                               dy,
                               self._strides,
                               self._padding)]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    conv_input, conv_filter = self.inputs
    if mesh_impl.tensor_dimension_to_mesh_axis(self._in_h_dim) is not None:
      raise ValueError("can't slice along dimension h")
    if mesh_impl.tensor_dimension_to_mesh_axis(self._in_w_dim) is not None:
      raise ValueError("can't slice along dimension w")
    if mesh_impl.tensor_dimension_to_mesh_axis(self._fh_dim) is not None:
      raise ValueError("can't slice along dimension fh")
    if mesh_impl.tensor_dimension_to_mesh_axis(self._fw_dim) is not None:
      raise ValueError("can't slice along dimension fw")
    def tf_fn(tf_input, tf_filter):
      output = tf.nn.conv2d(
          _tf_flatten_batch_dims(tf_input, 3),
          tf_filter, self._strides, self._padding)
      return _tf_restore_batch_dims(output, 3, tf_input)
    y = mesh_impl.slicewise(
        tf_fn, lowering.tensors[conv_input], lowering.tensors[conv_filter])
    # reducing out input channels - may need to allreduce
    in_mesh_axis = mesh_impl.tensor_dimension_to_mesh_axis(self._in_dim)
    # print("[conv2d lower]in_mesh_axis: ", in_mesh_axis)
    if in_mesh_axis is not None:
      def add_counter_fn():
        lowering.add_counter(
            "allreduce/%s/conv2d_op" % [in_mesh_axis],
            mesh_impl.laid_out_size(self.outputs[0].shape))
      y = LazyAllreduceSum(mesh_impl, y, [in_mesh_axis], add_counter_fn)
    # print("[conv2d lower]self.output[0]: ", self.outputs[0])
    # print("[conv2d lower]y: ", y.slice_shape)
    lowering.set_tensor_lowering(self.outputs[0], y)
    computation_shape = _shape_union([conv_filter.shape, self.outputs[0].shape])
    lowering.add_counter("conv2d/forward",
                         mesh_impl.laid_out_size(computation_shape))
    lowering.add_counter("conv2d_unique/forward", computation_shape.size)



def conv2d(x, output_dim, filter_size=(3, 3),
           strides=(1, 1), padding="SAME", filter_initializer=None,
           variable_dtype=None, name=None):
  """2D Convolution.

  Args:
    x: a mtf.Tensor of format NHWC.
    output_dim: a mtf.Dimension, indicating the output channel dimension.
    filter_size: a list or tuple in format [filter_height, filter_width].
    strides: a list or tuple in format [stride_height, stride_width].
    padding: either "SAME" or "VALID".
    filter_initializer: the initializer for tf.get_variable.
    variable_dtype: a mtf.VariableDType
    name: a string used for tf.variable_scope.

  Returns:
    a mtf.Tensor.
  """
  fh_dim = mtf.Dimension("fh", filter_size[0])
  fw_dim = mtf.Dimension("fw", filter_size[1])
  input_dim = x.shape[-1]
  with tf.variable_scope(name, default_name="conv2d"):
    if variable_dtype is None:
      variable_dtype = mtf.VariableDType(activation_dtype=x.dtype)
    conv_filter = mtf.get_variable(
        x.mesh, "kernel", [fh_dim, fw_dim, input_dim, output_dim],
        initializer=filter_initializer, dtype=variable_dtype)
    # Pad stride in batch and channel dimensions.
    strides = [1] + list(strides) + [1]

    return mtf.Conv2dOperation(x, conv_filter, strides, padding).outputs[0]
