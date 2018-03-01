import cv2
import numpy as np

from lib.config import cfg
from .chars_identify import chars_identify
from .chars_segment import CharsSegment


class CharsRecognize(object):
    def __init__(self):
        self.charsSegment = CharsSegment()

    def charsRecognize(self, plate, model_dir):
        chars = []
        result = self.charsSegment.charsSegment(plate, chars)
        if cfg.DEBUG:
            import matplotlib.pyplot as plt
            for i, char in enumerate(chars):
                plt.subplot(1, len(chars), i+1)
                plt.imshow(char, cmap='gray')
            plt.show()

        temp = []
        plate_license = ""

        if result == 0:
            temp = chars_identify.identify(np.array(chars)[..., None], model_dir)

        for index in temp:
            plate_license += cfg.CLASSES[index]

        return plate_license
