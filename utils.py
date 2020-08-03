"""Common utils."""
from typing import Text, Tuple, Union
import tensorflow as tf


def srelu_fn(x):
  """Smooth relu: a smooth version of relu."""
  with tf.name_scope('srelu'):
    beta = tf.Variable(20.0, name='srelu_beta', dtype=tf.float32)**2
    beta = tf.cast(beta, x.dtype)
    safe_log = tf.math.log(tf.where(x > 0., beta * x + 1., tf.ones_like(x)))
    return tf.where((x > 0.), x - (1. / beta) * safe_log, tf.zeros_like(x))


def activation_fn(features: tf.Tensor, act_type: Text):
  """Customized non-linear activation type."""
  if act_type == 'swish':
    return tf.nn.swish(features)
  elif act_type == 'swish_native':
    return features * tf.sigmoid(features)
  elif act_type == 'hswish':
    return features * tf.nn.relu6(features + 3) / 6
  elif act_type == 'relu':
    return tf.nn.relu(features)
  elif act_type == 'relu6':
    return tf.nn.relu6(features)
  elif act_type == 'mish':
    return features * tf.math.tanh(tf.math.softplus(features))
  elif act_type == 'srelu':
    return srelu_fn(features)
  else:
    raise ValueError('Unsupported act_type {}'.format(act_type))



def drop_connect(inputs, is_training, survival_prob):
  """Drop the entire conv with given survival probability."""
  # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
  if not is_training:
    return inputs

  # Compute tensor.
  batch_size = tf.shape(inputs)[0]
  random_tensor = survival_prob
  random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
  binary_tensor = tf.floor(random_tensor)
  # Unlike conventional way that multiply survival_prob at test time, here we
  # divide survival_prob at training time, such that no addition compute is
  # needed at test time.
  output = tf.div(inputs, survival_prob) * binary_tensor
  return output


conv_kernel_initializer = tf.initializers.variance_scaling()
dense_kernel_initializer = tf.initializers.variance_scaling()


# class Pair(tuple):
#
#   def __new__(cls, name, value):
#     return super(Pair, cls).__new__(cls, (name, value))
#
#   def __init__(self, name, _):  # pylint: disable=super-init-not-called
#     self.name = name

def parse_image_size(image_size: Union[Text, int, Tuple[int, int]]):
  """Parse the image size and return (height, width).
  Args:
    image_size: A integer, a tuple (H, W), or a string with HxW format.
  Returns:
    A tuple of integer (height, width).
  """
  if isinstance(image_size, int):
    # image_size is integer, with the same width and height.
    return (image_size, image_size)

  if isinstance(image_size, str):
    # image_size is a string with format WxH
    width, height = image_size.lower().split('x')
    return (int(height), int(width))

  if isinstance(image_size, tuple):
    return image_size

  raise ValueError('image_size must be an int, WxH string, or (height, width)'
                   'tuple. Was %r' % image_size)


def get_feat_sizes(image_size: Union[Text, int, Tuple[int, int]],
                   max_level: int):
  """Get feat widths and heights for all levels.
  Args:
    image_size: A integer, a tuple (H, W), or a string with HxW format.
    max_level: maximum feature level.
  Returns:
    feat_sizes: a list of tuples (height, width) for each level.
  """
  image_size = parse_image_size(image_size)
  feat_sizes = [{'height': image_size[0], 'width': image_size[1]}]
  feat_size = image_size
  for _ in range(1, max_level + 1):
    feat_size = ((feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1)
    feat_sizes.append({'height': feat_size[0], 'width': feat_size[1]})
  return feat_sizes


def verify_feats_size(feats,
                      feat_sizes,
                      min_level,
                      max_level,
                      data_format='channels_last'):
  """Verify the feature map sizes."""
  expected_output_size = feat_sizes[min_level:max_level + 1]
  for cnt, size in enumerate(expected_output_size):
    h_id, w_id = (2, 3) if data_format == 'channels_first' else (1, 2)
    if feats[cnt].shape[h_id] != size['height']:
      raise ValueError(
          'feats[{}] has shape {} but its height should be {}.'
          '(input_height: {}, min_level: {}, max_level: {}.)'.format(
              cnt, feats[cnt].shape, size['height'], feat_sizes[0]['height'],
              min_level, max_level))
    if feats[cnt].shape[w_id] != size['width']:
      raise ValueError(
          'feats[{}] has shape {} but its width should be {}.'
          '(input_width: {}, min_level: {}, max_level: {}.)'.format(
              cnt, feats[cnt].shape, size['width'], feat_sizes[0]['width'],
              min_level, max_level))
