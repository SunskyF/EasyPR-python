# @Time    : 2018/2/4
# @Author  : fh
# @File    : config.py
# @Desc    :
""" config system
This file specifies default config options for ULPR
It's learned from py-faster-rcnn
"""
import os
from pathlib import Path
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from lib.config import cfg
cfg = __C

__C.NAME = ""
#
# Detector
#
__C.DETECTOR = edict()

# The method to detect the plate
# 0: EasyPR method
# 1: Mask RCNN
__C.DETECTOR.METHOD = 0

# Detector Train Options
__C.DETECTOR.TRAIN = edict()

# Detector Test Options
__C.DETECTOR.TEST = edict()

#
# Recognizer
#
__C.RECOGNIZER = edict()

# The method to recognize the plate
# 0: EasyPR method
# 1: Multi-label
__C.RECOGNIZER.METHOD = 0

# Recognizer Train Options
__C.RECOGNIZER.TRAIN = edict()

# Multi-label
__C.RECOGNIZER.TRAIN.MULTILABEL = edict()
__C.RECOGNIZER.TRAIN.MULTILABEL.WEIGHT_DECAY = 0.0001
# Recognizer Test Options
__C.RECOGNIZER.TEST = edict()

#
# Solver
#
__C.SOLVER = edict()

# Train epoch
__C.SOLVER.EPOCH = 10

# Batch size
__C.SOLVER.BATCH_SIZE = 32

# Learning rate
__C.SOLVER.LEARNING_RATE = 0.01

#
# Train
#
# Train input parameters
__C.TRAIN = edict()

# Image width dimension
__C.TRAIN.IMAGE_WIDTH = 512

# Image height dimension
__C.TRAIN.IMAGE_HEIGHT = 512

# Display per iter
__C.TRAIN.DISPLAY = 20

# Summary per iter
__C.TRAIN.SUMMARY_INTERVAL = 200

# Flip image to augmentation
__C.TRAIN.FLIP = False

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0001

#
# Test
#
# Test input parameters
__C.TEST = edict()
# Image width dimension
__C.TEST.IMAGE_WIDTH = 512
# Image height dimension
__C.TEST.IMAGE_HEIGHT = 512

#
# MISC
#
# Root directory of project
__C.ROOT_DIR = Path(os.path.dirname(__file__), '..')

# Data directory
__C.DATA_DIR = __C.ROOT_DIR / 'data'

# Saved model
__C.OUTPUT_DIR = __C.ROOT_DIR / 'output'

# Use Detector
__C.USE_DETECTOR = True

# Use Recognizer
__C.USE_RECOGNIZER = True

# Debug Mode
__C.DEBUG = False

# Visualize in intermediate process
__C.VIS = False

# Plate char dict
__C.CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
               'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
               'W', 'X', 'Y', 'Z', '川', '鄂', '赣', '甘', '贵',
               '桂', '黑', '沪', '冀', '津', '京', '吉', '辽',
               '鲁', '蒙', '闽', '宁', '青', '琼', '陕', '苏',
               '晋', '皖', '湘', '新', '豫', '渝', '粤', '云',
               '藏', '浙')  # 65 classes

# Image mean (RGB)
__C.PIXEL_MEAN = np.array([[[122.7717, 115.9465, 102.9801]]])


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
    assert __C.NAME != "", "Please add a name"
