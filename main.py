"""The main training script."""
import os
import tensorflow as tf
import numpy as np
from absl import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG)
import dataloader
import det_model_fn
import hparams_config
import utils


class _Flags:
    pass

## eliminated TPU and distributed training code to make this all easier to digest for noobs
FLAGS = _Flags()
FLAGS.model_dir = None      # Location of model_dir
FLAGS.backbone_ckpt = ''    # Location of the ResNet50 checkpoint to use for model initialization.
FLAGS.ckpt = None           # Start training from this EfficientDet checkpoint.
FLAGS.train_batch_size = 64             # training batch size
FLAGS.eval_batch_size = 1               # evaluation batch size
FLAGS.eval_samples = 5000               # The number of samples for evaluation.
FLAGS.training_file_pattern = None      # REQUIRED: Glob for training data files (e.g., COCO train - minival set)
FLAGS.validation_file_pattern = None    # REQUIRED: Glob for evaluation tfrecords (e.g., COCO val2017 set)')
FLAGS.val_json_file = None              # COCO validation JSON containing golden bounding boxes. If None, use the ground
                                        # truth from the dataloader. Ignored if testdev_dir is not None.
FLAGS.testdev_dir = None                # COCO testdev dir. If not None, ignore val_json_file.
FLAGS.num_examples_per_epoch = 120000   # Number of examples in one epoch
FLAGS.num_epochs = 5000                 # Number of epochs for training.  Default = 300.
FLAGS.mode = 'train'                    # Mode to run: train or eval (default: train)
FLAGS.model_name = 'efficientdet-d0'    # Model name.
FLAGS.eval_after_training = False       # Run one eval after the training finishes.
FLAGS.tf_random_seed = None             # Sets the TF graph seed for deterministic execution across runs (for debugging).
# Comma separated k=v pairs of hyperparameters or a module containing attributes to use as hyperparameters
# image_size options = 512 + φ · 128
# act_type = activation_fn in utils
# set momentum to 0 since BN is updated by averaging over entire train set
# iou loss: ('iou', 'ciou', 'diou', 'giou')
# Questions:  What is anchor scale?  Is "momentum" for batch norm?
FLAGS.hparams = {'act_type': 'swish', 'skip_mismatch': False, 'backbone_name': 'efficientnet-b0', 'name': FLAGS.model_name,
                 'separable_conv': True, 'image_size': 512, 'momentum': 0, 'box_loss_weight': None, 'iou_loss_type': 'ciou',
                 'iou_loss_weight': 1.0, 'learning_rate': 0.008}

# For Eval mode
FLAGS.min_eval_interval = 180           # Minimum seconds between evaluations.
FLAGS.eval_timeout = None               # Maximum seconds between checkpoints before evaluation terminates.


def main(_):
    ## no tpus or distributed training.  Just local GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Check data path
    if FLAGS.mode in ('train', 'train_and_eval') and FLAGS.training_file_pattern is None:
        raise RuntimeError('You must specify --training_file_pattern for training.')
    if FLAGS.mode in ('eval', 'train_and_eval'):
        if FLAGS.validation_file_pattern is None:
            raise RuntimeError('You must specify --validation_file_pattern for evaluation.')

    # Parse and override hparams
    config = hparams_config.get_detection_config(FLAGS.model_name)
    config.override(FLAGS.hparams)
    if FLAGS.num_epochs:  # NOTE: remove this flag after updating all docs.
        config.num_epochs = FLAGS.num_epochs

    # Parse image size in case it is in string format.
    config.image_size = utils.parse_image_size(config.image_size)

    params = dict(
        config.as_dict(),
        model_name=FLAGS.model_name,
        model_dir=FLAGS.model_dir,
        num_examples_per_epoch=FLAGS.num_examples_per_epoch,
        backbone_ckpt=FLAGS.backbone_ckpt,
        ckpt=FLAGS.ckpt,
        val_json_file=FLAGS.val_json_file,
        testdev_dir=FLAGS.testdev_dir,
        mode=FLAGS.mode)

    model_dir = FLAGS.model_dir

    model_fn_instance = det_model_fn.get_model_fn(FLAGS.model_name)
    max_instances_per_image = config.max_instances_per_image




if __name__ == '__main__':
    main()