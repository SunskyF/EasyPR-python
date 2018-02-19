import datetime
import tensorflow as tf
from pathlib import Path


class Train(object):
    def __init__(self, params):
        self.learning_rate = params['lr']
        self.number_epoch = params['number_epoch']
        self.epoch_length = params['epoch_length']
        now = datetime.datetime.now()
        self.log_dir = Path('{}_{:%Y%m%dT%H%M}'.format(params['log_dir'], now))

        print("lr: {}, number_epochs: {}, epoch_length: {}, max_steps: {}".format(
            self.learning_rate, self.number_epoch, self.epoch_length, int(self.epoch_length * self.number_epoch)
        ))

        self.model = None

        self.pred_labels = None
        self.loss = None
        self.total_loss = None
        self.l2_loss = None

        self.sess = None

        self.weights = []
        self.biases = []

    def compile(self, model):
        self.model = model

        self.pred_logits = self.model.pred_logits
        self.pred_labels = self.model.pred_labels
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.pred_logits, labels=self.model.y))
        tf.summary.scalar('loss', self.loss)

        self.total_loss = self.loss + 5e-4 * self.model.l2_loss

        correct_pred = tf.equal(self.pred_labels, self.model.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)
        self.__add_optimal()

    def __add_optimal(self):
        optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        train_op = optimizer.minimize(self.total_loss)

        self.train_op = train_op

    def train(self, train_dataset, val_dataset):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(str(self.log_dir / "summary" / "train"), self.sess.graph)
        test_writer = tf.summary.FileWriter(str(self.log_dir / "summary" / "test"))
        model_dir = self.log_dir / 'models'

        saver = tf.train.Saver(max_to_keep=3)
        tf.train.write_graph(self.sess.graph_def, model_dir, 'model.pb', False)
        self.sess.run(tf.global_variables_initializer())

        for step in range(int(self.epoch_length * self.number_epoch)):

            train_x, train_y = train_dataset.batch()
            feed_dict = {self.model.x: train_x, self.model.y: train_y, self.model.keep_prob: 0.5}
            self.sess.run(self.train_op, feed_dict=feed_dict)

            if step % 10 == 0:
                feed_dict = {self.model.x: train_x, self.model.y: train_y, self.model.keep_prob: 1}
                summary, train_loss, acc = self.sess.run([merged, self.loss, self.accuracy], feed_dict=feed_dict)
                print("Step: %d / %d (epoch: %d / %d), Train_loss: %g, acc: %g" % (step % self.epoch_length,
                                                                                   self.epoch_length,
                                                                                   step // self.epoch_length,
                                                                                   self.number_epoch, train_loss, acc))
                train_writer.add_summary(summary, step)

            if step % 100 == 0:
                val_x, val_y = val_dataset.batch()
                feed_dict = {self.model.x: val_x, self.model.y: val_y, self.model.keep_prob: 1}
                summary, valid_loss, acc = self.sess.run([merged, self.loss, self.accuracy], feed_dict=feed_dict)
                print("%s ---> Validation_loss: %g, acc: %g" % (datetime.datetime.now(), valid_loss, acc))
                test_writer.add_summary(summary, step)

            if step % self.epoch_length == self.epoch_length - 1:
                now_epoch = step // self.epoch_length
                print('Saving checkpoint: ', now_epoch)
                saver.save(self.sess, str(model_dir / "model.ckpt"), now_epoch)

        train_writer.close()
        test_writer.close()
        self.close()

    def close(self):
        self.sess.close()


def eval_model(nodes, samples_feed, eval_sess=None, model_dir=None, first=True):
    if first:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver()
            print("Model restored...", ckpt.model_checkpoint_path)
            saver.restore(eval_sess, ckpt.model_checkpoint_path)
    return eval_sess.run(nodes, samples_feed)
