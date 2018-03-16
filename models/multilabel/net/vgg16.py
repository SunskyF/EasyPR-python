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

from .network import MultiLabel


class Vgg16(MultiLabel):
    def __init__(self, cfg):
        super(Vgg16, self).__init__(cfg)
        self._scope = 'vgg_16'

    def _backbone(self, reuse=None):
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3],
                              trainable=False, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                              trainable=False, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                              scope='conv3')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                              scope='conv4')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                              scope='conv5')

        self._layers['head'] = net

        return net

    def _tail(self, net_head, reuse=None):
        with tf.variable_scope(self._scope, self._scope, reuse=None):
            flat = slim.flatten(net_head, scope='flatten')
            fc6 = slim.fully_connected(flat, 1024, scope='fc6')
            fc7 = slim.fully_connected(fc6, 1024, scope='fc7')
        return fc7

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            if len(v.name.split('/')) > 1 and v.name.split('/')[1] in ['fc6', 'fc7']:
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)
        return variables_to_restore
