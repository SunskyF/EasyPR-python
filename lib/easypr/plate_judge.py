import numpy as np
import tensorflow as tf

from models.easypr.net.judgenet import Judgenet
from models.easypr.cnn_train import eval_model


class _PlateJudge:
    def __init__(self):
        self.graph = tf.Graph()
        self.eval_sess = None
        self.model = None

    def judge(self, plates, model_dir):
        judgeRes = []
        images = []
        for plate in plates:
            images.append(plate.plate_image)

        tmp = np.array(images) / 255 * 2 - 1
        first = self.eval_sess is None

        if first:
            # if first load
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.eval_sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            if first:
                self.model = Judgenet()
                self.model.compile()
            pred = eval_model(self.model.pred_labels,
                              {self.model.x: tmp, self.model.keep_prob: 1},
                              model_dir=model_dir,
                              eval_sess=self.eval_sess, first=first)

        for i, judge_res in enumerate(pred):
            if judge_res:
                judgeRes.append(plates[i])

        return judgeRes


plate_judge = _PlateJudge()
