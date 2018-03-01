# @Time    : 2018/2/12
# @Author  : fh
# @File    : plate_detect.py
# @Desc    :
"""
    Use mask-rcnn to do plate detect
"""
import cv2
import keras.backend.tensorflow_backend as KTF
import numpy as np
import tensorflow as tf

import models.mrcnn.model as modellib
from lib.config import cfg
from models.mrcnn.plate import PlateInferenceConfig


class Singleton(type):
    _inst = {}

    def __call__(self, *args, **kw):
        if self not in self._inst:
            self._inst[self] = super(Singleton, self).__call__(*args, **kw)
        return self._inst[self]


class PlateDetect(metaclass=Singleton):
    def __init__(self):
        self.graph = tf.Graph()
        self.eval_sess = None
        self.model = None

    def detect(self, src, model_dir):
        first = self.eval_sess is None
        if first:
            # if first load
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.eval_sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            if first:
                plate_config = PlateInferenceConfig()
                # plate_config.display()
                self.model = modellib.MaskRCNN(mode="inference", config=plate_config, model_dir=model_dir)
            KTF.set_session(self.eval_sess)
            model_path = self.model.find_last()[1]
            self.model.load_weights(model_path, by_name=True)
            result = self.model.detect([src])
        return self._post_process(result, src)

    def _post_process(self, result, src=None):
        masks = result[0]['masks']
        bboxes = result[0]['rois']
        plates = []
        for idx, bbox in enumerate(bboxes):
            mask = masks[..., idx]
            x1, y1, x2, y2 = min(bbox[1], bbox[3]), min(bbox[2], bbox[0]), max(bbox[1], bbox[3]), max(bbox[2], bbox[0])
            """
                (x1, y1) -------- (x2, y1)
                        |        |
                        |        |
                        |        |
                (x1, y2) -------- (x2, y2)        
            """
            w, h = abs(bbox[3] - bbox[1]), abs(bbox[2] - bbox[0])
            # use mask.copy() because of a bug of opencv-python for some pictures
            # TypeError: Layout of the output array image is incompatible with cv::Mat (step[ndims-1] != elemsize
            # or step[1] != elemsize*nchannels)
            _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0].squeeze()
            dis_11 = np.linalg.norm(contour - np.array([x1, y1]), axis=1)
            top_left = contour[np.argmin(dis_11)] + np.array([-3, -3])
            dis_12 = np.linalg.norm(contour - np.array([x1, y2]), axis=1)
            bottom_left = contour[np.argmin(dis_12)] + np.array([-3, 3])
            dis_21 = np.linalg.norm(contour - np.array([x2, y1]), axis=1)
            top_right = contour[np.argmin(dis_21)] + np.array([3, -3])
            dis_22 = np.linalg.norm(contour - np.array([x2, y2]), axis=1)
            bottom_right = contour[np.argmin(dis_22)] + np.array([3, 3])
            approx = np.vstack([top_left, bottom_left, bottom_right, top_right])
            plates.append(approx)

            if cfg.VIS:
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                ax1.set_title("Output mask")
                ax1.imshow(src)
                ax1.add_patch(
                    patches.Rectangle((x1, y1), w, h, fill=False, linewidth=1, edgecolor="red")
                )
                ax1.add_patch(patches.Polygon(contour))
                ax2 = fig.add_subplot(212)
                ax2.set_title("Post-process mask")
                ax2.imshow(src)
                ax2.add_patch(
                    patches.Rectangle((x1, y1), w, h, fill=False, linewidth=1, edgecolor="red")
                )
                ax2.add_patch(patches.Polygon(approx))
                plt.show()
        return plates
