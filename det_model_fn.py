"""Model function definition, including both architecture and loss."""
import functools
import re
import numpy as np
import tensorflow as tf

import coco_metric
import efficientdet_arch
import hparams_config
import iou_utils
import nms_np
import utils
from keras import anchors
from keras import postprocess


def focal_loss(y_pred, y_true, alpha, gamma, normalizer, label_smoothing=0.0):
  """Compute the focal loss between `logits` and the golden `target` values.
  Focal loss = -(1-pt)^gamma * log(pt)
  where pt is the probability of being classified to the true class.
  Args:
    y_pred: A float32 tensor of size [batch, height_in, width_in,
      num_predictions].
    y_true: A float32 tensor of size [batch, height_in, width_in,
      num_predictions].
    alpha: A float32 scalar multiplying alpha to the loss from positive examples
      and (1-alpha) to the loss from negative examples.
    gamma: A float32 scalar modulating loss from hard and easy examples.
    normalizer: Divide loss by this value.
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.
  Returns:
    loss: A float32 scalar representing normalized total loss.
  """
  with tf.name_scope('focal_loss'):
    alpha = tf.convert_to_tensor(alpha, dtype=y_pred.dtype)
    gamma = tf.convert_to_tensor(gamma, dtype=y_pred.dtype)

    # compute focal loss multipliers before label smoothing, such that it will
    # not blow up the loss.
    pred_prob = tf.sigmoid(y_pred)
    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t) ** gamma

    # apply label smoothing for cross_entropy for each entry.
    y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    # compute the final loss and return
    return (1 / normalizer) * alpha_factor * modulating_factor * ce

def _box_loss(box_outputs, box_targets, num_positives, delta=0.1):
  """Computes box regression loss."""
  # delta is typically around the mean value of regression target.
  # for instances, the regression targets of 512x512 input with 6 anchors on
  # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
  normalizer = num_positives * 4.0
  mask = tf.not_equal(box_targets, 0.0)
  box_loss = tf.losses.huber(
      box_targets,
      box_outputs,
      weights=mask,
      delta=delta,
      reduction=tf.losses.Reduction.SUM)
  box_loss /= normalizer
  return box_loss


def _box_iou_loss(box_outputs, box_targets, num_positives, iou_loss_type):
  """Computes box iou loss."""
  normalizer = num_positives * 4.0
  box_iou_loss = iou_utils.iou_loss(box_outputs, box_targets, iou_loss_type)
  box_iou_loss = tf.reduce_sum(box_iou_loss) / normalizer
  return box_iou_loss


def detection_loss(cls_outputs, box_outputs, labels, params):
  """Computes total detection loss.
  Computes total detection loss including box and class loss from all levels.
  Args:
    cls_outputs: an OrderDict with keys representing levels and values
      representing logits in [batch_size, height, width, num_anchors].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in [batch_size, height, width,
      num_anchors * 4].
    labels: the dictionary that returned from dataloader that includes
      groundtruth targets.
    params: the dictionary including training parameters specified in
      default_haprams function in this file.
  Returns:
    total_loss: an integer tensor representing total loss reducing from
      class and box losses from all levels.
    cls_loss: an integer tensor representing total class loss.
    box_loss: an integer tensor representing total box regression loss.
    box_iou_loss: an integer tensor representing total box iou loss.
  """
  # Sum all positives in a batch for normalization and avoid zero
  # num_positives_sum, which would lead to inf loss during training
  num_positives_sum = tf.reduce_sum(labels['mean_num_positives']) + 1.0
  levels = cls_outputs.keys()

  cls_losses = []
  box_losses = []
  for level in levels:
    # Onehot encoding for classification labels.
    cls_targets_at_level = tf.one_hot(labels['cls_targets_%d' % level],
                                      params['num_classes'])

    if params['data_format'] == 'channels_first':
      bs, _, width, height, _ = cls_targets_at_level.get_shape().as_list()
      cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                        [bs, -1, width, height])
    else:
      bs, width, height, _, _ = cls_targets_at_level.get_shape().as_list()
      cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                        [bs, width, height, -1])
    box_targets_at_level = labels['box_targets_%d' % level]

    cls_loss = focal_loss(
        cls_outputs[level],
        cls_targets_at_level,
        params['alpha'],
        params['gamma'],
        normalizer=num_positives_sum,
        label_smoothing=params['label_smoothing'])

    if params['data_format'] == 'channels_first':
      cls_loss = tf.reshape(cls_loss,
                            [bs, -1, width, height, params['num_classes']])
    else:
      cls_loss = tf.reshape(cls_loss,
                            [bs, width, height, -1, params['num_classes']])
    cls_loss *= tf.cast(
        tf.expand_dims(tf.not_equal(labels['cls_targets_%d' % level], -2), -1),
        tf.float32)
    cls_losses.append(tf.reduce_sum(cls_loss))

    if params['box_loss_weight']:
      box_losses.append(
          _box_loss(
              box_outputs[level],
              box_targets_at_level,
              num_positives_sum,
              delta=params['delta']))

  if params['iou_loss_type']:
    input_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                    params['num_scales'],
                                    params['aspect_ratios'],
                                    params['anchor_scale'],
                                    params['image_size'])
    box_output_list = [tf.reshape(box_outputs[i], [-1, 4]) for i in levels]
    box_outputs = tf.concat(box_output_list, axis=0)
    box_target_list = [
        tf.reshape(labels['box_targets_%d' % level], [-1, 4])
        for level in levels
    ]
    box_targets = tf.concat(box_target_list, axis=0)
    anchor_boxes = tf.tile(input_anchors.boxes, [params['batch_size'], 1])
    box_outputs = anchors.decode_box_outputs(box_outputs, anchor_boxes)
    box_targets = anchors.decode_box_outputs(box_targets, anchor_boxes)
    box_iou_loss = _box_iou_loss(box_outputs, box_targets, num_positives_sum, params['iou_loss_type'])

  else:
    box_iou_loss = 0

  # Sum per level losses to total loss.
  cls_loss = tf.add_n(cls_losses)
  box_loss = tf.add_n(box_losses) if box_losses else 0

  total_loss = (
      cls_loss +
      params['box_loss_weight'] * box_loss +
      params['iou_loss_weight'] * box_iou_loss)

  return total_loss, cls_loss, box_loss, box_iou_loss


def reg_l2_loss(model, weight_decay, regex=r'.*(kernel|weight):0$'):
  """Return regularization l2 loss loss."""
  var_match = re.compile(regex)
  return weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables() if var_match.match(v.name)])


def _model_fn(features, labels, mode, params, model, variable_filter_fn=None):
  """Model definition entry.
  Args:
    features: the input image tensor with shape [batch_size, height, width, 3].
      The height and width are fixed and equal.
    labels: the input labels in a dictionary. The labels include class targets
      and box targets which are dense label maps. The labels are generated from
      get_input_fn function in data/dataloader.py
    mode: the mode of TPUEstimator including TRAIN, EVAL, and PREDICT.
    params: the dictionary defines hyperparameters of model. The default
      settings are in default_hparams function in this file.
    model: the model outputs class logits and box regression outputs.
    variable_filter_fn: the filter function that takes trainable_variables and
      returns the variable list after applying the filter rule.
  Returns:
    tpu_spec: the TPUEstimatorSpec to run training, evaluation, or prediction.
  Raises:
    RuntimeError: if both ckpt and backbone_ckpt are set.
  """
  tf.summary.image('input_image', features, global_step)
  training_hooks = []

  def _model_outputs(inputs):
    # Convert params (dict) to Config for easier access.
    return model(inputs, config=hparams_config.Config(params))

  precision = utils.get_precision(params['strategy'], params['mixed_precision'])
  cls_outputs, box_outputs = utils.build_model_with_precision(
      precision, _model_outputs, features, params['is_training_bn'])

  levels = cls_outputs.keys()
  for level in levels:
    cls_outputs[level] = tf.cast(cls_outputs[level], tf.float32)
    box_outputs[level] = tf.cast(box_outputs[level], tf.float32)

  # First check if it is in PREDICT mode.
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'image': features,
    }
    for level in levels:
      predictions['cls_outputs_%d' % level] = cls_outputs[level]
      predictions['box_outputs_%d' % level] = box_outputs[level]
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Set up training loss and learning rate.

  learning_rate = params['learning_rate']

  # cls_loss and box_loss are for logging. only total_loss is optimized.
  det_loss, cls_loss, box_loss, box_iou_loss = detection_loss(cls_outputs, box_outputs, labels, params)
  reg_l2loss = reg_l2_loss(params['weight_decay'])
  total_loss = det_loss + reg_l2loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    if params['optimizer'].lower() == 'sgd':
      optimizer = tf.optimizers.SGD(learning_rate, momentum=params['momentum'])
    elif params['optimizer'].lower() == 'adam':
      optimizer = tf.optimizers.Adam(learning_rate)
    else:
      raise ValueError('optimizers should be adam or sgd')

    # filter out frozen vars?
    var_list = model.trainable_variables()
    if variable_filter_fn:
      var_list = variable_filter_fn(var_list)

    # get batch norm variables
    # ........................
    for _ in range(params['num_epochs']):
      # train loop below
      with tf.GradientTape(persistent=False) as tape:
          mdl_full_out = model_full(x_full, training=True)
      grads = tape.gradient(loss, model_parms_grouped)
      if params.get('clip_gradients_norm', 0) > 0:
          grads, gnorm = tf.clip_by_global_norm(grads, params['clip_gradients_norm'])
      global_step = optimizer.apply_gradients(zip(grads, model_parms_grouped))

      tf.summary.scalar('gnorm', gnorm)
      tf.summary.scalar('trainloss/cls_loss', cls_loss, global_step)
      if params['box_loss']:
          tf.summary.scalar('trainloss/box_loss', box_loss, global_step)
      if params['iou_loss_type']:
          tf.summary.scalar('trainloss/box_iou_loss', box_iou_loss)
      tf.summary.scalar('trainloss/det_loss', det_loss, global_step)
      tf.summary.scalar('trainloss/reg_l2_loss', reg_l2loss, global_step)
      tf.summary.scalar('trainloss/loss', total_loss, global_step)


  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:

    def metric_fn(**kwargs):
      """Returns a dictionary that has the evaluation metrics."""
      if params['nms_configs'].get('pyfunc', True):
        detections_bs = []
        for index in range(kwargs['boxes'].shape[0]):
          nms_configs = params['nms_configs']
          detections = tf.numpy_function(
              functools.partial(nms_np.per_class_nms, nms_configs=nms_configs),
              [
                  kwargs['boxes'][index],
                  kwargs['scores'][index],
                  kwargs['classes'][index],
                  tf.slice(kwargs['image_ids'], [index], [1]),
                  tf.slice(kwargs['image_scales'], [index], [1]),
                  params['num_classes'],
                  nms_configs['max_output_size'],
              ], tf.float32)
          detections_bs.append(detections)
      else:
        # These two branches should be equivalent, but currently they are not.
        # TODO(tanmingxing): enable the non_pyfun path after bug fix.
        nms_boxes, nms_scores, nms_classes, _ = postprocess.per_class_nms(
            params, kwargs['boxes'], kwargs['scores'], kwargs['classes'],
            kwargs['image_scales'])
        img_ids = tf.cast(
            tf.expand_dims(kwargs['image_ids'], -1), nms_scores.dtype)
        detections_bs = [
            img_ids * tf.ones_like(nms_scores),
            nms_boxes[:, :, 1],
            nms_boxes[:, :, 0],
            nms_boxes[:, :, 3] - nms_boxes[:, :, 1],
            nms_boxes[:, :, 2] - nms_boxes[:, :, 0],
            nms_scores,
            nms_classes,
        ]
        detections_bs = tf.stack(detections_bs, axis=-1, name='detnections')

      if params.get('testdev_dir', None):
        # logging.info('Eval testdev_dir %s', params['testdev_dir'])
        eval_metric = coco_metric.EvaluationMetric(
            testdev_dir=params['testdev_dir'])
        coco_metrics = eval_metric.estimator_metric_fn(detections_bs,
                                                       tf.zeros([1]))
      else:
        # logging.info('Eval val with groudtruths %s.', params['val_json_file'])
        eval_metric = coco_metric.EvaluationMetric(
            filename=params['val_json_file'])
        coco_metrics = eval_metric.estimator_metric_fn(
            detections_bs, kwargs['groundtruth_data'])

      # Add metrics to output.
      cls_loss = tf.metrics.mean(kwargs['cls_loss_repeat'])
      box_loss = tf.metrics.mean(kwargs['box_loss_repeat'])
      output_metrics = {
          'cls_loss': cls_loss,
          'box_loss': box_loss,
      }
      output_metrics.update(coco_metrics)
      return output_metrics

    cls_loss_repeat = tf.reshape(tf.tile(tf.expand_dims(cls_loss, 0), [params['batch_size'],]), [params['batch_size'], 1])
    box_loss_repeat = tf.reshape(tf.tile(tf.expand_dims(box_loss, 0), [params['batch_size'],]), [params['batch_size'], 1])

    cls_outputs = postprocess.to_list(cls_outputs)
    box_outputs = postprocess.to_list(box_outputs)
    params['nms_configs']['max_nms_inputs'] = anchors.MAX_DETECTION_POINTS
    boxes, scores, classes = postprocess.pre_nms(params, cls_outputs, box_outputs)
    metric_fn_inputs = {
        'cls_loss_repeat': cls_loss_repeat,
        'box_loss_repeat': box_loss_repeat,
        'image_ids': labels['source_ids'],
        'groundtruth_data': labels['groundtruth_data'],
        'image_scales': labels['image_scales'],
        'boxes': boxes,
        'scores': scores,
        'classes': classes,
    }
    eval_metrics = (metric_fn, metric_fn_inputs)

  checkpoint = params.get('ckpt') or params.get('backbone_ckpt')

  if checkpoint and mode == tf.estimator.ModeKeys.TRAIN:
    # Initialize the model from an EfficientDet or backbone checkpoint.
    if params.get('ckpt') and params.get('backbone_ckpt'):
      raise RuntimeError('--backbone_ckpt and --checkpoint are mutually exclusive')

    if params.get('backbone_ckpt'):
      var_scope = params['backbone_name'] + '/'
      if params['ckpt_var_scope'] is None:
        # Use backbone name as default checkpoint scope.
        ckpt_scope = params['backbone_name'] + '/'
      else:
        ckpt_scope = params['ckpt_var_scope'] + '/'
    else:
      # Load every var in the given checkpoint
      var_scope = ckpt_scope = '/'

    def scaffold_fn():
      """Loads pretrained model through scaffold function."""
      # logging.info('restore variables from %s', checkpoint)
      var_map = utils.get_ckpt_var_map(
          ckpt_path=checkpoint,
          ckpt_scope=ckpt_scope,
          var_scope=var_scope,
          skip_mismatch=params['skip_mismatch'])

      tf.train.init_from_checkpoint(checkpoint, var_map)

      return tf.train.Scaffold()
  elif mode == tf.estimator.ModeKeys.EVAL and moving_average_decay:

    def scaffold_fn():
      """Load moving average variables for eval."""
      logging.info('Load EMA vars with ema_decay=%f', moving_average_decay)
      restore_vars_dict = ema.variables_to_restore(ema_vars)
      saver = tf.train.Saver(restore_vars_dict)
      return tf.train.Scaffold(saver=saver)
  else:
    scaffold_fn = None


  return None


def efficientdet_model_fn(features, labels, mode, params):
  """EfficientDet model."""
  variable_filter_fn = functools.partial(
      efficientdet_arch.freeze_vars, pattern=params['var_freeze_expr'])
  return _model_fn(
      features,
      labels,
      mode,
      params,
      model=efficientdet_arch.efficientdet,
      variable_filter_fn=variable_filter_fn)


def get_model_arch(model_name='efficientdet-d0'):
  """Get model architecture for a given model name."""
  if 'efficientdet' in model_name:
    return efficientdet_arch.efficientdet

  raise ValueError('Invalide model name {}'.format(model_name))


def get_model_fn(model_name='efficientdet-d0'):
  """Get model fn for a given model name."""
  if 'efficientdet' in model_name:
    return efficientdet_model_fn

  raise ValueError('Invalide model name {}'.format(model_name))