# -*- coding: utf-8 -*-
"""Layers that construct a LeNet-CNN network.

Basic layers:
- Input image
- Input label
- Convolution layer
- Activation layer
- Max pooling layer
- Flatten layer (flatten one image)
- Dense layer
- Output layer

Composite layers:
- ConvPool layer (conv + activation + maxpool)
"""

import tensorflow as tf


class Layer(object):
    """Layer base class

    - Base class has only input and output.
    - Base class keeps an ID that is also total number of layers"""
    id = 0

    def __init__(self, layer_in, layer_out, weights=[], biases=[]):
        self.output = layer_out
        self.input = layer_in
        self.weights = weights
        self.biases = biases


class ImageLayer(Layer):
    """Input image layer"""

    def __init__(self, image_height, image_width, n_channels, layer_name='IMAGE'):
        with tf.variable_scope(layer_name):
            image = tf.placeholder(tf.float32, [None, image_height, image_width, n_channels], 'images')

        super().__init__(layer_in=None, layer_out=image)


class LabelLayer(Layer):
    """Input label layer"""

    def __init__(self, layer_name='LABEL'):
        with tf.variable_scope(layer_name):
            label = tf.placeholder(tf.int64, [None], 'labels')

        super().__init__(layer_in=None, layer_out=label)


class ConvLayer(Layer):
    """2D-Convolution layer"""

    def __init__(self,
                 prev_layer,
                 filter_height, filter_width, n_filters,
                 layer_name='CONV2D',
                 stride=(1, 1),
                 padding='SAME',
                 weight_init=tf.contrib.layers.xavier_initializer(),
                 bias_init=tf.constant_initializer(0.0)):
        n_input_channels = prev_layer.output.get_shape().as_list()[3]

        with tf.variable_scope(layer_name):
            filters = tf.get_variable('kernel',
                                      shape=[filter_height, filter_width, n_input_channels, n_filters],
                                      initializer=weight_init)
            biases = tf.get_variable('biases',
                                     shape=[n_filters],
                                     initializer=bias_init)
            layer_output = tf.nn.bias_add(
                tf.nn.conv2d(input=prev_layer.output,
                             filter=filters,
                             strides=[1, stride[0], stride[1], 1],
                             padding=padding),
                biases)

        super().__init__(layer_in=prev_layer.output, layer_out=layer_output, weights=[filters, ], biases=[biases, ])


class ActivationLayer(Layer):
    """Activation layer"""

    def __init__(self, prev_layer, activation=tf.nn.relu, layer_name='ACTIVATION'):
        with tf.variable_scope(layer_name):
            layer_output = activation(prev_layer.output)

        super().__init__(layer_in=prev_layer.output, layer_out=layer_output)


class MaxPoolLayer(Layer):
    """Max pooling layer"""

    def __init__(self, prev_layer, pool_height, pool_width, layer_name='MAXPOOL', padding='SAME'):
        with tf.variable_scope(layer_name):
            layer_output = tf.nn.max_pool(prev_layer.output,
                                          ksize=[1, pool_height, pool_width, 1],
                                          strides=[1, pool_height, pool_width, 1],
                                          padding=padding)

        super().__init__(layer_in=prev_layer.output, layer_out=layer_output)


class FlattenLayer(Layer):
    """Flatten layer of one image"""

    def __init__(self, prev_layer, layer_name='FLATTEN'):
        shape = prev_layer.output.get_shape().as_list()
        with tf.variable_scope(layer_name):
            layer_output = tf.reshape(prev_layer.output, [-1, shape[1] * shape[2] * shape[3]])

        super().__init__(layer_in=prev_layer.output, layer_out=layer_output)


class DenseLayer(Layer):
    """Dense layer"""

    def __init__(self,
                 prev_layer,
                 n_out,
                 layer_name='DENSE',
                 weight_init=tf.contrib.layers.xavier_initializer(),
                 bias_init=tf.constant_initializer(0.0)):
        n_in = prev_layer.output.get_shape()[1].value

        with tf.variable_scope(layer_name):
            weights = tf.get_variable('weights', shape=[n_in, n_out], initializer=weight_init)
            biases = tf.get_variable('biases', shape=[n_out], initializer=bias_init)
            layer_output = tf.matmul(prev_layer.output, weights) + biases

        super().__init__(layer_in=prev_layer.output, layer_out=layer_output, weights=[weights, ], biases=[biases, ])


class OutputLayer(Layer):
    """Output layer"""

    def __init__(self,
                 prev_layer,
                 n_out,
                 layer_name='OUTPUT',
                 weight_init=tf.contrib.layers.xavier_initializer(),
                 bias_init=tf.constant_initializer(0.0)):
        n_in = prev_layer.output.get_shape()[1].value

        with tf.variable_scope(layer_name):
            weights = tf.get_variable('weights', shape=[n_in, n_out], initializer=weight_init)
            biases = tf.get_variable('biases', shape=[n_out], initializer=bias_init)
            layer_output = tf.matmul(prev_layer.output, weights) + biases

        super().__init__(layer_in=prev_layer.output, layer_out=layer_output, weights=[weights, ], biases=[biases, ])


class DropoutLayer(Layer):
    def __init__(self, prev_layer, prob, layer_name='dropout'):
        with tf.name_scope(layer_name):
            layer_output = tf.nn.dropout(prev_layer.output, prob)
        super().__init__(layer_in=prev_layer.output, layer_out=layer_output)


class ConvPoolLayer(Layer):
    """Convolution + Activation + MaxPool"""

    def __init__(self,
                 prev_layer,
                 filter_height, filter_width, n_filters,
                 pool_height, pool_width,
                 layer_name='',
                 stride=(1, 1),
                 padding='SAME',
                 weight_init=tf.contrib.layers.xavier_initializer(),
                 bias_init=tf.constant_initializer(0.0),
                 activation=tf.nn.relu):
        with tf.variable_scope("CONVPOOL" + '_' + layer_name):
            conv = ConvLayer(prev_layer,
                             filter_height, filter_width, n_filters, "CONV" + '_' + layer_name,
                             stride, padding, weight_init, bias_init)
            act = ActivationLayer(conv, activation, layer_name='CPACTIVATION' + '_' + layer_name)
            pool = MaxPoolLayer(act, pool_height, pool_width, "POOL" + '_' + layer_name, padding)

        self.conv = conv
        self.activation = act
        self.pool = pool
        super().__init__(layer_in=prev_layer.output, layer_out=pool.output, weights=conv.weights, biases=conv.biases)
