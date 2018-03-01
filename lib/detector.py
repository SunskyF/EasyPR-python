# @Time    : 2018/2/7
# @Author  : fh
# @File    : detector.py
# @Desc    :
"""
    The detector wrapper of ULPR. It will detect the plate roi of a raw image.
"""

import cv2

from lib.config import cfg
# lib methods
from lib.easypr.plate_detect import PlateDetect as easypr_detect
from lib.mrcnn.plate_detect import PlateDetect as mrcnn_detect
from lib.utils.find_last import find_last

_methods = [
    'easypr',  # 0
    'mrcnn',   # 1
]


def detect(src):
    """
        Detector factory
    :param src: raw image
    :return: [[x1, y1, x2, y2, x3, y3, x4, y4]]
    """
    method = cfg.DETECTOR.METHOD
    return eval('_detect_' + _methods[method])(src)


# method 0
def _detect_easypr(src):
    pd = easypr_detect()
    pd.setPDLifemode(True)
    dir_name = find_last(cfg.OUTPUT_DIR, 'whether_car')
    return pd.plateDetect(src, str(cfg.OUTPUT_DIR / dir_name / 'models'))


def _detect_mrcnn(src):
    src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
    pd = mrcnn_detect()
    return pd.detect(src, str(cfg.OUTPUT_DIR))
