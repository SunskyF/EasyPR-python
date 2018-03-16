# @Time    : 2018/3/5
# @Author  : fh
# @File    : train.py
# @Desc    :
"""
    Multi-label train solver wrapper
"""
import datetime

import tensorflow as tf
import numpy as np
import os

from .net.vgg16 import Vgg16 as vgg16
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow


class MultiLabelSolver:
    def __init__(self, net, cfg, train_dataset, val_dataset, pretrained_model=None):
        assert net in ['vgg16'], 'The net is not supported.'
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.net = eval(net)(self.cfg)
        self.pretrained_model = pretrained_model
        self.log_dir = os.path.join(self.cfg.OUTPUT_DIR,
                                    '{}_{:%Y%m%dT%H%M}'.format(self.cfg.NAME, datetime.datetime.now()))
        self.summary_dir = os.path.join(self.log_dir, 'summary', 'train')
        self.val_summary_dir = os.path.join(self.log_dir, 'summary', 'val')
        self.model_dir = os.path.join(self.log_dir, 'model')

    def train(self, ):
        train_dataset_generator = self.train_dataset.next_batch()
        val_dataset_generator = self.val_dataset.next_batch()

        lr, train_op = self.construct_graph()
        self.initialize()

        epoch = self.cfg.SOLVER.EPOCH
        self.sess.run(tf.assign(lr, self.cfg.SOLVER.LEARNING_RATE))

        total_iter = 0
        for i in range(epoch):
            max_iters = len(self.train_dataset.image_ids) // self.cfg.SOLVER.BATCH_SIZE
            for iter in range(len(self.train_dataset.image_ids) // self.cfg.SOLVER.BATCH_SIZE):
                train_images, train_labels = next(train_dataset_generator)
                blob = {'image': train_images, 'label': train_labels}
                total_loss = self.net.train_step(self.sess, blob, train_op)

                if iter % self.cfg.TRAIN.DISPLAY == 0:
                    print(
                        'iter: {} / {} epoch: {} / {}, total loss: {}\n'.format(iter, max_iters, i, epoch, total_loss))
                if iter % self.cfg.TRAIN.SUMMARY_INTERVAL == 0:
                    # TODO: add summary
                    pass

                total_iter += 1
            filename = os.path.join(self.model_dir, self.cfg.NAME + '_iter_{:d}'.format(total_iter) + '.ckpt')
            self.saver.save(self.sess, filename)
            print('Wrote snapshot to: {:s}'.format(filename))
            self.writer.close()
            self.val_writer.close()

    def initialize(self):
        # Fresh train directly from ImageNet weights
        print('Loading initial model weights from {:s}'.format(self.pretrained_model))
        variables = tf.global_variables()
        # Initialize all variables first
        self.sess.run(tf.variables_initializer(variables, name='init'))
        var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
        # Get the variables to restore, ignoring the variables to fix
        variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)

        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(self.sess, self.pretrained_model)
        print('Loaded.')

    def construct_graph(self):
        with self.sess.graph.as_default():
            layers = self.net.complie('train', self.train_dataset.num_classes)
            loss = layers['total_loss']
            lr = tf.Variable(self.cfg.SOLVER.LEARNING_RATE, trainable=False)
            self.optimizer = tf.train.AdamOptimizer(lr)
            train_op = self.optimizer.minimize(loss)

            self.writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
            self.val_writer = tf.summary.FileWriter(self.val_summary_dir)
            self.saver = tf.train.Saver()
        return lr, train_op

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")
