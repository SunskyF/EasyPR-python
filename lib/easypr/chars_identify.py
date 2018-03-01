import tensorflow as tf

from models.easypr.net.lenet import Lenet
from models.easypr.cnn_train import eval_model


class _CharsIdentify:
    def __init__(self):
        self.graph = tf.Graph()
        self.eval_sess = None
        self.model = None

    def identify(self, images, model_dir):
        tmp = images / 255 * 2 - 1
        first = self.eval_sess is None

        if first:
            # if first load
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.eval_sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            if first:
                self.model = Lenet()
                self.model.compile()
            pred = eval_model(self.model.pred_labels,
                              {self.model.x: tmp, self.model.keep_prob: 1},
                              model_dir=model_dir,
                              eval_sess=self.eval_sess, first=first)

        return pred


chars_identify = _CharsIdentify()
