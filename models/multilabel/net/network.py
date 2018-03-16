# @Time    : 2018/3/2
# @Author  : fh
# @File    : network.py
# @Desc    :
"""
    The basic network class, now used in multilabel training only
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

from .utils import draw


class MultiLabel:
    def __init__(self, cfg):
        self.cfg = cfg
        self._predictions = {}
        self._losses = {}
        self._layers = {}

        # summary
        self._gt_image = None
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}

    def complie(self, mode, num_classes):
        self._image = tf.placeholder(tf.float32,
                                     shape=[None, self.cfg.TRAIN.IMAGE_HEIGHT, self.cfg.TRAIN.IMAGE_WIDTH, 3])
        self._label = tf.placeholder(tf.int32, shape=[None, 7])

        self._mode = mode
        self._num_classes = num_classes

        weights_regularizer = tf.contrib.layers.l2_regularizer(self.cfg.TRAIN.WEIGHT_DECAY)
        biases_regularizer = tf.no_regularizer

        with arg_scope([slim.conv2d, slim.conv2d_in_plane, slim.conv2d_transpose,
                        slim.separable_conv2d, slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=biases_regularizer,
                       biases_initializer=tf.constant_initializer(0.0)):
            self._build_network()

        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        if self._mode == 'train':
            self._add_loss()
            self._predictions.update(self._losses)

            val_summaries = []

            # add summary
            with tf.device("/cpu:0"):
                val_summaries.append(self._add_gt_image_summary())
                for key, var in self._event_summaries.items():
                    val_summaries.append(tf.summary.scalar(key, var))
                for key, var in self._score_summaries.items():
                    self._add_score_summary(key, var)
                for var in self._train_summaries:
                    self._add_train_summary(var)

            self._summary_op = tf.summary.merge_all()
            self._summary_op_val = tf.summary.merge(val_summaries)
        return self._predictions

    def _add_loss(self):
        with tf.variable_scope('LOSS_') as scope:
            cls_scores = self._predictions["cls_score"]
            label = self._label
            for i in range(self._label.shape[1]):
                self._losses['loss_label' + str(i)] = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_scores[i], labels=label[:, i]))
                tf.add_to_collection('losses', self._losses['loss_label' + str(i)])
            loss = tf.add_n(tf.get_collection('losses'))
            regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
            self._losses['total_loss'] = loss + regularization_loss

            self._event_summaries.update(self._losses)
        return loss

    def _build_network(self):
        net_head = self._backbone()
        fc7 = self._tail(net_head)
        with tf.variable_scope(self._scope, self._scope):
            self._multilabel_classification(fc7)
        self._score_summaries.update(self._predictions['cls_score'])

    def _backbone(self, reuse=None):
        raise NotImplementedError

    def _tail(self, net_head, reuse=None):
        raise NotImplementedError

    def _multilabel_classification(self, net):
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)

        cls_scores = []
        cls_probs = []
        cls_preds = []
        for i in range(self._label.shape[1]):
            cls_score = slim.fully_connected(net, self._num_classes,
                                             weights_initializer=initializer, scope='cls_score' + str(i))
            cls_prob = slim.softmax(cls_score, scope="cls_prob" + str(i))
            cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred" + str(i))
            cls_scores.append(cls_score)
            cls_probs.append(cls_prob)
            cls_preds.append(cls_pred)

        self._predictions['cls_score'] = dict(zip(range(len(cls_scores)), cls_scores))
        self._predictions['cls_prob'] = cls_probs
        self._predictions['cls_pred'] = cls_preds

    def train_step(self, sess, blob, train_op):
        feed_dict = {self._image: blob['image'], self._label: blob['label']}
        loss_0, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss, _ = sess.run([
            self._losses['loss_label0'],
            self._losses['loss_label1'],
            self._losses['loss_label2'],
            self._losses['loss_label3'],
            self._losses['loss_label4'],
            self._losses['loss_label5'],
            self._losses['loss_label6'],
            self._losses['total_loss'],
            train_op],
            feed_dict=feed_dict)
        return loss_0, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss

    def train_step_with_summary(self, sess, blob, train_op):
        feed_dict = {self._image: blob['image'], self._label: blob['label']}
        loss_0, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss, summary, _ = sess.run([
            self._losses['loss_label0'],
            self._losses['loss_label1'],
            self._losses['loss_label2'],
            self._losses['loss_label3'],
            self._losses['loss_label4'],
            self._losses['loss_label5'],
            self._losses['loss_label6'],
            self._losses['total_loss'],
            self._summary_op,
            train_op],
            feed_dict=feed_dict)
        return loss_0, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss, summary

    def get_summary(self, sess, blob):
        feed_dict = {self._image: blob['image'],
                     self._label: blob['label']}
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

        return summary

    def test_image(self, sess, image):
        feed_dict = {self._image: image}
        key_pred = sess.run([self._predictions["keypoint_pred"]], feed_dict=feed_dict)

        return key_pred

    def get_variables_to_restore(self, variables, var_keep_dic):
        raise NotImplementedError

    def fix_variables(self, sess, pretrained_model):
        raise NotImplementedError

    # summary
    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + str(key) + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def _add_gt_image(self):
        # add back mean
        image = self._image + self.cfg.PIXEL_MEAN
        # BGR to RGB (opencv uses BGR)
        self._gt_image = tf.image.resize_bilinear(image, [self.cfg.TEST.IMAGE_HEIGHT, self.cfg.TEST.IMAGE_WIDTH])

    def _add_gt_image_summary(self):
        # use a customized visualization function to visualize the boxes
        if self._gt_image is None:
            self._add_gt_image()
        image = tf.py_func(draw,
                           [self._gt_image, self._label],
                           tf.float32, name="gt_keypoints")

        return tf.summary.image('GROUND_TRUTH', image)
