"""Common utils."""
import contextlib
import os
import re
from typing import Text, Tuple, Union
import numpy as np
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


### UPDATE:  this needs to be updated for eager execution (currently for graph mode)
### change to method used in previous works for updating BN each epoch
def get_ema_vars():
  """Get all exponential moving average (ema) variables."""
  ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
  for v in tf.global_variables():
    # We maintain mva for batch norm moving mean and variance as well.
    if 'moving_mean' in v.name or 'moving_variance' in v.name:
      ema_vars.append(v)
  return list(set(ema_vars))


def get_ckpt_var_map(ckpt_path, ckpt_scope, var_scope, skip_mismatch=None):
  """Get a var map for restoring from pretrained checkpoints.
  Args:
    ckpt_path: string. A pretrained checkpoint path.
    ckpt_scope: string. Scope name for checkpoint variables.
    var_scope: string. Scope name for model variables.
    skip_mismatch: skip variables if shape mismatch.
  Returns:
    var_map: a dictionary from checkpoint name to model variables.
  """
  logging.info('Init model from checkpoint {}'.format(ckpt_path))
  if not ckpt_scope.endswith('/') or not var_scope.endswith('/'):
    raise ValueError('Please specific scope name ending with /')
  if ckpt_scope.startswith('/'):
    ckpt_scope = ckpt_scope[1:]
  if var_scope.startswith('/'):
    var_scope = var_scope[1:]

  var_map = {}
  # Get the list of vars to restore.
  model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=var_scope)
  reader = tf.train.load_checkpoint(ckpt_path)
  ckpt_var_name_to_shape = reader.get_variable_to_shape_map()
  ckpt_var_names = set(reader.get_variable_to_shape_map().keys())

  for i, v in enumerate(model_vars):
    if not v.op.name.startswith(var_scope):
      logging.info('skip {} -- does not match scope {}'.format(
          v.op.name, var_scope))
    ckpt_var = ckpt_scope + v.op.name[len(var_scope):]
    if (ckpt_var not in ckpt_var_names and
        v.op.name.endswith('/ExponentialMovingAverage')):
      ckpt_var = ckpt_scope + v.op.name[:-len('/ExponentialMovingAverage')]

    if ckpt_var not in ckpt_var_names:
      if 'Momentum' in ckpt_var or 'RMSProp' in ckpt_var:
        # Skip optimizer variables.
        continue
      if skip_mismatch:
        logging.info('skip {} ({}) -- not in ckpt'.format(v.op.name, ckpt_var))
        continue
      raise ValueError('{} is not in ckpt {}'.format(v.op, ckpt_path))

    if v.shape != ckpt_var_name_to_shape[ckpt_var]:
      if skip_mismatch:
        logging.info('skip {} ({} vs {}) -- shape mismatch'.format(
            v.op.name, v.shape, ckpt_var_name_to_shape[ckpt_var]))
        continue
      raise ValueError('shape mismatch {} ({} vs {})'.format(
          v.op.name, v.shape, ckpt_var_name_to_shape[ckpt_var]))

    if i < 5:
      # Log the first few elements for sanity check.
      logging.info('Init {} from ckpt var {}'.format(v.op.name, ckpt_var))
    var_map[ckpt_var] = v

  return var_map


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


def num_params_flops(readable_format=True):
  """Return number of parameters and flops."""
  nparams = np.sum(
      [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
  options = tf.profiler.ProfileOptionBuilder.float_operation()
  options['output'] = 'none'
  flops = tf.profiler.profile(
      tf.get_default_graph(), options=options).total_float_ops
  # We use flops to denote multiply-adds, which is counted as 2 ops in tfprof.
  flops = flops // 2
  if readable_format:
    nparams = float(nparams) * 1e-6
    flops = float(flops) * 1e-9
  return nparams, flops


conv_kernel_initializer = tf.initializers.variance_scaling()
dense_kernel_initializer = tf.initializers.variance_scaling()


class Pair(tuple):

  def __new__(cls, name, value):
    return super(Pair, cls).__new__(cls, (name, value))

  def __init__(self, name, _):  # pylint: disable=super-init-not-called
    self.name = name

def archive_ckpt(ckpt_eval, ckpt_objective, ckpt_path):
  """Archive a checkpoint if the metric is better."""
  ckpt_dir, ckpt_name = os.path.split(ckpt_path)

  saved_objective_path = os.path.join(ckpt_dir, 'best_objective.txt')
  saved_objective = float('-inf')
  if tf.io.gfile.exists(saved_objective_path):
    with tf.io.gfile.GFile(saved_objective_path, 'r') as f:
      saved_objective = float(f.read())
  if saved_objective > ckpt_objective:
    logging.info('Ckpt {} is worse than {}'.format(ckpt_objective,
                                                   saved_objective))
    return False

  filenames = tf.io.gfile.glob(ckpt_path + '.*')
  if filenames is None:
    logging.info('No files to copy for checkpoint {}'.format(ckpt_path))
    return False

  # clear up the backup folder.
  backup_dir = os.path.join(ckpt_dir, 'backup')
  if tf.io.gfile.exists(backup_dir):
    tf.io.gfile.rmtree(backup_dir)

  # rename the old checkpoints to backup folder.
  dst_dir = os.path.join(ckpt_dir, 'archive')
  if tf.io.gfile.exists(dst_dir):
    logging.info('mv {} to {}'.format(dst_dir, backup_dir))
    tf.io.gfile.rename(dst_dir, backup_dir)

  # Write checkpoints.
  tf.io.gfile.makedirs(dst_dir)
  for f in filenames:
    dest = os.path.join(dst_dir, os.path.basename(f))
    tf.io.gfile.copy(f, dest, overwrite=True)
  ckpt_state = tf.train.generate_checkpoint_state_proto(
      dst_dir,
      model_checkpoint_path=os.path.join(dst_dir, ckpt_name))
  with tf.io.gfile.GFile(os.path.join(dst_dir, 'checkpoint'), 'w') as f:
    f.write(str(ckpt_state))
  with tf.io.gfile.GFile(os.path.join(dst_dir, 'best_eval.txt'), 'w') as f:
    f.write('%s' % ckpt_eval)

  # Update the best objective.
  with tf.io.gfile.GFile(saved_objective_path, 'w') as f:
    f.write('%f' % ckpt_objective)

  logging.info('Copying checkpoint {} to {}'.format(ckpt_path, dst_dir))
  return True


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


def get_precision(strategy: str, mixed_precision: bool = False):
  """Get the precision policy for a given strategy."""
  if mixed_precision:
    if strategy == 'tpu':
      return 'mixed_bfloat16'

    if tf.config.experimental.list_physical_devices('GPU'):
      return 'mixed_float16'

    # TODO(fsx950223): Fix CPU float16 inference
    # https://github.com/google/automl/issues/504
    logging.warning('float16 is not supported for CPU, use float32 instead')
    return 'float32'

  return 'float32'


@contextlib.contextmanager
def float16_scope():
  """Scope class for float16."""

  def _custom_getter(getter, *args, **kwargs):
    """Returns a custom getter that methods must be called under."""
    cast_to_float16 = False
    requested_dtype = kwargs['dtype']
    if requested_dtype == tf.float16:
      kwargs['dtype'] = tf.float32
      cast_to_float16 = True
    var = getter(*args, **kwargs)
    if cast_to_float16:
      var = tf.cast(var, tf.float16)
    return var

  with tf.variable_scope('', custom_getter=_custom_getter) as varscope:
    yield varscope


def set_precision_policy(policy_name: Text = None, loss_scale: bool = False):
  """Set precision policy according to the name.
  Args:
    policy_name: precision policy name, one of 'float32', 'mixed_float16',
      'mixed_bfloat16', or None.
    loss_scale: whether to use loss scale (only for training).
  """
  if not policy_name:
    return

  assert policy_name in ('mixed_float16', 'mixed_bfloat16', 'float32')
  logging.info('use mixed precision policy name %s', policy_name)
  # TODO(tanmingxing): use tf.keras.layers.enable_v2_dtype_behavior() when it
  # available in stable TF release.
  from tensorflow.python.keras.engine import base_layer_utils  # pylint: disable=g-import-not-at-top,g-direct-tensorflow-import
  base_layer_utils.enable_v2_dtype_behavior()
  # mixed_float16 training is not supported for now, so disable loss_scale.
  # float32 and mixed_bfloat16 do not need loss scale for training.
  if loss_scale:
    policy = tf2.keras.mixed_precision.experimental.Policy(policy_name)
  else:
    policy = tf2.keras.mixed_precision.experimental.Policy(
        policy_name, loss_scale=None)
  tf2.keras.mixed_precision.experimental.set_policy(policy)


def build_model_with_precision(pp, mm, ii, tt, *args, **kwargs):
  """Build model with its inputs/params for a specified precision context.
  This is highly specific to this codebase, and not intended to be general API.
  Advanced users only. DO NOT use it if you don't know what it does.
  NOTE: short argument names are intended to avoid conficts with kwargs.
  Args:
    pp: A string, precision policy name, such as "mixed_float16".
    mm: A function, for rmodel builder.
    ii: A tensor, for model inputs.
    tt: A bool, If true, it is for training; otherwise, it is for eval.
    *args: A list of model arguments.
    **kwargs: A dict, extra model parameters.
  Returns:
    the output of mm model.
  """
  if pp == 'mixed_bfloat16':
    set_precision_policy(pp)
    inputs = tf.cast(ii, tf.bfloat16)
    with tf.tpu.bfloat16_scope():
      outputs = mm(inputs, *args, **kwargs)
    set_precision_policy('float32')
  elif pp == 'mixed_float16':
    set_precision_policy(pp, loss_scale=tt)
    inputs = tf.cast(ii, tf.float16)
    with float16_scope():
      outputs = mm(inputs, *args, **kwargs)
    set_precision_policy('float32')
  elif not pp or pp == 'float32':
    outputs = mm(ii, *args, **kwargs)
  else:
    raise ValueError('Unknow precision name {}'.format(pp))

  # Users are responsible to convert the dtype of all outputs.
  return outputs