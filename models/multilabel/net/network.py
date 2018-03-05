# @Time    : 2018/3/2
# @Author  : fh
# @File    : network.py
# @Desc    :
"""
    The basic network class, now used in multilabel training only
"""
import tensorflow as tf


class MultiLabel:
    def __init__(self, cfg):
        self.cfg = cfg
        self._predictions = {}
        self._losses = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._layers = {}
        self._gt_image = None
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self._variables_to_fix = {}

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def complie(self, mode, num_classes):
        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])

        self._mode = mode
        self._num_classew = num_classes

        weights_regularizer = tf.contrib.layers.l2_regularizer(self.cfg.TRAIN.WEIGHT_DECAY)
        biases_regularizer = tf.no_regularizer

    def _backbone(self, is_training, reuse=None):
        raise NotImplementedError

    def get_summary(self, sess, blobs):
        feed_dict = {self._image: blobs['data']}
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

        return summary

    def train_step(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data']}
        loss, _ = sess.run([self._losses['cross_entropy'],
                            train_op],
                           feed_dict=feed_dict)
        return loss

    def train_step_with_summary(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data']}
        loss, summary, _ = sess.run([self._losses['cross_entropy'],
                                     self._summary_op,
                                     train_op],
                                    feed_dict=feed_dict)
        return loss, summary

    def train_step_no_return(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        sess.run([train_op], feed_dict=feed_dict)
