from lib.utils.nms import py_cpu_nms
from lib.utils.sort_points import sort_points
from .core_func import *
from .plate_judge import plate_judge
from .plate_locate import PlateLocate


class PlateDetect(object):
    def __init__(self):
        self.m_plateLocate = PlateLocate()
        self.m_maxPlates = 3

    def setPDLifemode(self, param):
        self.m_plateLocate.setLifemode(param)

    def plateDetect(self, src, model_dir):
        res = []
        color_plates = []
        sobel_plates = []

        self.m_plateLocate.plateColorLocate(src, color_plates)

        if len(color_plates) != 0:
            res = plate_judge.judge(color_plates, model_dir)

        if len(res) <= self.m_maxPlates:
            self.m_plateLocate.plateSobelLocate(src, sobel_plates)

            if len(sobel_plates) != 0:
                res += plate_judge.judge(sobel_plates, model_dir)

        boxes = []
        for det in res:
            boxes.append(cv2.boundingRect(cv2.boxPoints(det.plate_pos)))
        boxes = np.array(boxes).reshape((-1, 4))
        dets = np.hstack((boxes, np.ones((boxes.shape[0], 1))))
        keep = py_cpu_nms(dets, 0.2)
        result = []
        for idx, det in enumerate(res):
            if keep[idx]:
                result.append(sort_points(cv2.boxPoints(det.plate_pos)))
        return result




