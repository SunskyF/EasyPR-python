# @Time    : 2018/3/2
# @Author  : fh
# @File    : vgg16.py
# @Desc    :
"""
    VGG16 implement in tensorflow
    Referenced from https://github.com/endernewton/tf-faster-rcnn/blob/master/lib/nets/vgg16.py
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim

from models.multilabel.net.network import MultiLabel


class Vgg16(MultiLabel):
    def __init__(self):
        MultiLabel.__init__(self)
        self._scope = 'vgg_16'

    def _backbone(self, is_training, reuse=None):
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3],
                              trainable=False, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                              trainable=False, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                              trainable=is_training, scope='conv3')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                              trainable=is_training, scope='conv4')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                              trainable=is_training, scope='conv5')

        self._act_summaries.append(net)
        self._layers['head'] = net

        return net