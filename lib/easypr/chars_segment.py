import cv2
import numpy as np

from lib.easypr.core_func import getPlateType, Color, ThresholdOtsu, clearLiuDingChar


class CharsSegment(object):
    def __init__(self):
        self.LiuDingSize = 7
        self.MatWidth = 136

        self.colorThreshold = 150
        self.BluePercent = 0.3
        self.WhitePercent = 0.1

        self.m_debug = True

    def verifyCharSizes(self, r):
        aspect = 0.5
        charAspect = r.shape[1] / r.shape[0]
        error = 0.7
        minH = 10
        maxH = 35

        minAspect = 0.05  # for number 1
        maxAspect = aspect + aspect * error

        area = cv2.countNonZero(r)
        bbArea = r.shape[0] * r.shape[1]
        percPixels = area / bbArea

        if percPixels <= 1 and minAspect < charAspect < maxAspect and minH <= r.shape[0] < maxH:
            return True
        else:
            return False

    def preprocessChar(self, in_mat):
        h = in_mat.shape[0]
        w = in_mat.shape[1]

        charSize = 20
        transform = np.array([[1, 0, 0],
                              [0, 1, 0]], dtype=np.float32)
        m = max(w, h)
        transform[0][2] = m / 2 - w / 2
        transform[1][2] = m / 2 - h / 2

        warpImage = cv2.warpAffine(in_mat, transform, (m, m), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)

        return cv2.resize(warpImage, (charSize, charSize))

    def charsSegment(self, input, result):
        w = input.shape[1]
        h = input.shape[0]

        tmp = input[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]

        plateType = getPlateType(tmp, True)

        input_gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

        if plateType == Color.BLUE:

            w = input_gray.shape[1]
            h = input_gray.shape[0]

            tmp = input_gray[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]

            threadHoldV = ThresholdOtsu(tmp)

            _, img_threshold = cv2.threshold(input_gray, threadHoldV, 255, cv2.THRESH_BINARY)
        elif plateType == Color.YELLOW:
            w = input_gray.shape[1]
            h = input_gray.shape[0]

            tmp = input_gray[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]

            threadHoldV = ThresholdOtsu(tmp)

            _, img_threshold = cv2.threshold(input_gray, threadHoldV, 255, cv2.THRESH_BINARY_INV)
        elif plateType == Color.WHITE:
            _, img_threshold = cv2.threshold(input_gray, 10, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        else:
            _, img_threshold = cv2.threshold(input_gray, 10, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

        if not clearLiuDingChar(img_threshold):
            return 2

        img_contours = img_threshold.copy()

        _, contours, _ = cv2.findContours(img_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        vecRect = []
        for it in contours:

            mr = cv2.boundingRect(it)
            x, y, w, h = map(int, mr)

            roi = img_threshold[y:y + h, x:x + w]
            if self.verifyCharSizes(roi):
                vecRect.append(mr)

        if len(vecRect) == 0:
            return 3

        vecRect = sorted(vecRect, key=lambda x: x[0])

        specIndex = self.GetSpecificRect(vecRect)

        if specIndex < len(vecRect):
            chineseRect = self.GetChineseRect(vecRect[specIndex])
        else:
            return 4

        newSorted = []
        newSorted.append(chineseRect)
        self.RebuildRect(vecRect, newSorted, specIndex)
        if len(newSorted) == 0:
            return 5

        for mr in newSorted:
            x, y, w, h = map(int, mr)
            auxRoi = input_gray[y:y + h, x:x + w]
            if plateType == Color.BLUE:
                _, newroi = cv2.threshold(auxRoi, 5, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif plateType == Color.YELLOW:
                _, newroi = cv2.threshold(auxRoi, 5, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            elif plateType == Color.WHITE:
                _, newroi = cv2.threshold(auxRoi, 5, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
            else:
                _, newroi = cv2.threshold(auxRoi, 5, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

            newroi = self.preprocessChar(newroi)

            result.append(newroi)
        return 0

    def GetSpecificRect(self, vecRect):
        xpos = []
        maxH = 0
        maxW = 0
        for rect in vecRect:  # (x, y, w, h)
            xpos.append(rect[0])

            if rect[3] > maxH:
                maxH = rect[3]
            if rect[2] > maxW:
                maxW = rect[2]
        specIndex = 0
        for i in range(len(vecRect)):
            mr = vecRect[i]
            midx = mr[0] + mr[2] / 2
            if (mr[2] > maxW * 0.8 or mr[3] > maxH * 0.8) and \
                                    int(self.MatWidth / 7) < midx < 2 * int(self.MatWidth / 7):
                specIndex = i
        return specIndex

    def GetChineseRect(self, rectSpe):
        h = rectSpe[3]
        newW = rectSpe[2] * 1.15
        x = rectSpe[0]
        y = rectSpe[1]

        newX = x - int(newW * 1.15)
        newX = newX if newX > 0 else 0

        return (newX, y, int(newW), h)

    def RebuildRect(self, vecRect, outRect, specIndex):
        count = 6
        for i in range(specIndex, len(vecRect)):
            if count == 0:
                break
            outRect.append(vecRect[i])
            count -= 1
