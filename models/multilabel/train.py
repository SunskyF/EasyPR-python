# @Time    : 2018/3/5
# @Author  : fh
# @File    : train.py
# @Desc    :
"""
    Multi-label train solver wrapper
"""
import tensorflow as tf
import numpy as np

from .net.vgg16 import Vgg16


class MultiLabelSolver:
    def __init__(self, net, cfg):
        assert net in ['vgg16'], 'The net is not supported.'
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        self.session = tf.Session(config=tfconfig)
