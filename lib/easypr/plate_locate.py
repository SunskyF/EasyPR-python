import cv2

from .base import Plate
from .core_func import *


class PlateLocate(object):
    def __init__(self):
        self.m_GaussianBlurSize = 5
        self.m_MorphSizeWidth = 17
        self.m_MorphSizeHeight = 3

        self.m_error = 0.9
        self.m_aspect = 3.75
        self.m_verifyMin = 1
        self.m_verifyMax = 24
        self.m_angle = 60

    def setLifemode(self, param):
        if param:
            self.m_GaussianBlurSize = 5
            self.m_MorphSizeWidth = 10
            self.m_MorphSizeHeight = 3
            self.m_error = 0.75
            self.m_aspect = 4
            self.m_verifyMin = 1
            self.m_verifyMax = 200
        else:
            self.m_GaussianBlurSize = 5
            self.m_MorphSizeWidth = 17
            self.m_MorphSizeHeight = 3
            self.m_error = 0.9
            self.m_aspect = 3.75
            self.m_verifyMin = 1
            self.m_verifyMax = 24

    def plateSobelLocate(self, src, cand_plates):
        bound_rects = self.sobelFrtSearch(src)
        bound_rects_part = []

        for i in range(len(bound_rects)):
            fRatio = bound_rects[i][2] / bound_rects[i][3]

            if fRatio < 3.0 and fRatio > 1.0 and bound_rects[i][3] < 120:
                itemRect = bound_rects[i]  # [x, y, w, h]

                itemRect[0] = itemRect[0] - itemRect[3] * (4 - fRatio)
                if (itemRect[0] < 0):
                    itemRect[0] = 0

                itemRect[2] = itemRect[2] + itemRect[3] * 2 * (4 - fRatio)

                if (itemRect[2] + itemRect[0] >= src.shape[1]):
                    itemRect[2] = src.shape[1] - itemRect[0]

                itemRect[1] = itemRect[1] - itemRect[3] * 0.08
                itemRect[3] = itemRect[3] * 1.16
                bound_rects_part.append(itemRect)
        rects_sobel = []
        for i in range(len(bound_rects_part)):
            bound_rect = bound_rects_part[i]

            refpoint = (bound_rect[0], bound_rect[1])
            x = int(bound_rect[0] if bound_rect[0] > 0 else 0)
            y = int(bound_rect[1] if bound_rect[1] > 0 else 0)

            width = int(bound_rect[2] if x + bound_rect[2] < src.shape[1] else src.shape[1] - x)
            height = int(bound_rect[3] if y + bound_rect[3] < src.shape[0] else src.shape[0] - y)

            bound_mat = src[y: y + height, x: x + width, :]

            self.sobelSecSearchPart(bound_mat, refpoint, rects_sobel)
        for i in range(len(bound_rects)):
            bound_rect = bound_rects[i]
            refpoint = (bound_rect[0], bound_rect[1])
            x = int(bound_rect[0] if bound_rect[0] > 0 else 0)
            y = int(bound_rect[1] if bound_rect[1] > 0 else 0)

            width = int(bound_rect[2] if x + bound_rect[2] < src.shape[1] else src.shape[1] - x)
            height = int(bound_rect[3] if y + bound_rect[3] < src.shape[0] else src.shape[0] - y)

            bound_mat = src[y: y + height, x: x + width, :]

            self.sobelSecSearch(bound_mat, refpoint, rects_sobel)

        src_b = self.sobelOper(src, 3, 10, 3)
        self.deskew(src, src_b, rects_sobel, cand_plates)

    def sobelSecSearchPart(self, bound, refpoint, out):
        bound_threshold = self.sobelOperT(bound, 3, 6, 2)

        tempBoundThread = bound_threshold.copy()
        clearLiuDingOnly(tempBoundThread)

        posLeft, posRight, flag = bFindLeftRightBound(tempBoundThread)
        if flag:
            if posRight != 0 and posLeft != 0 and posLeft < posRight:
                posY = int(bound_threshold.shape[0] * 0.5)
                for i in range(posLeft + int(bound_threshold.shape[0] * 0.1), posRight - 4):
                    bound_threshold[posY, i] = 255
            for i in range(bound_threshold.shape[0]):
                bound_threshold[i, posLeft] = 0
                bound_threshold[i, posRight] = 0

        _, contours, _ = cv2.findContours(bound_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for it in contours:
            mr = cv2.minAreaRect(it)
            if self.verifySizes(mr):
                tmp = (mr[0][0] + refpoint[0], mr[0][1] + refpoint[1])
                out.append((tmp, mr[1], mr[2]))

    def sobelSecSearch(self, bound, refpoint, out):
        bound_threshold = self.sobelOper(bound, 3, 10, 3)
        _, contours, _ = cv2.findContours(bound_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for it in contours:
            mr = cv2.minAreaRect(it)
            if self.verifySizes(mr):
                tmp = (mr[0][0] + refpoint[0], mr[0][1] + refpoint[1])
                out.append((tmp, mr[1], mr[2]))

    def sobelFrtSearch(self, src):
        out_rects = []

        src_threshold = self.sobelOper(src, self.m_GaussianBlurSize, self.m_MorphSizeWidth, self.m_MorphSizeHeight)
        _, contours, _ = cv2.findContours(src_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for it in contours:
            mr = cv2.minAreaRect(it)

            if self.verifySizes(mr):
                safeBoundRect, flag = self.calcSafeRect(mr, src)
                if not flag:
                    continue
                out_rects.append(safeBoundRect)

        return out_rects

    def deskew(self, src, src_b, inRect, outRect):
        m_angle = 60

        for i in range(len(inRect)):
            roi_rect = inRect[i]

            r = roi_rect[1][0] / roi_rect[1][1]  # width / height
            roi_angle = roi_rect[2]
            roi_rect_size = roi_rect[1]

            if r < 1:
                roi_angle += 90
                roi_rect_size = (roi_rect[1][1], roi_rect[1][0])  # h, w

            if -m_angle < roi_angle < m_angle:
                (x, y, w, h), flag = self.calcSafeRect(roi_rect, src)
                if not flag:
                    continue
                bound_mat = src[y: y + h, x: x + w, :]
                bound_mat_b = src_b[y: y + h, x: x + w]

                roi_ref_center = (roi_rect[0][0] - x, roi_rect[0][1] - y)

                if (-3 < roi_angle < 3) or roi_angle == 90 or roi_angle == -90:
                    deskew_mat = bound_mat
                else:
                    # TODO test
                    rotated_mat, flag = self.rotation(bound_mat, roi_rect_size, roi_ref_center,
                                                      roi_angle)
                    if not flag:
                        continue

                    rotated_mat_b, flag = self.rotation(bound_mat_b, roi_rect_size, roi_ref_center,
                                                        roi_angle)
                    if not flag:
                        continue

                    roi_slope, flag = self.isdeflection(rotated_mat_b, roi_angle)
                    if flag:
                        deskew_mat = self.affine(rotated_mat, roi_slope)
                    else:
                        deskew_mat = rotated_mat

                self.DeleteNotArea(deskew_mat)

                if (deskew_mat.shape[1] / deskew_mat.shape[0]) > 2.3 and (
                            deskew_mat.shape[1] / deskew_mat.shape[0]) < 6:
                    if deskew_mat.shape[0] >= 36 or deskew_mat.shape[1] >= 136:
                        plate_mat = cv2.resize(deskew_mat, (136, 36), interpolation=cv2.INTER_AREA)
                    else:
                        plate_mat = cv2.resize(deskew_mat, (136, 36), interpolation=cv2.INTER_CUBIC)
                    plate = Plate()
                    plate.plate_pos = roi_rect
                    plate.plate_image = plate_mat
                    outRect.append(plate)

    def sobelOper(self, img, blursize, morphW, morphH):
        blur = cv2.GaussianBlur(img, (blursize, blursize), 0, 0, cv2.BORDER_DEFAULT)

        if len(blur.shape) == 3:
            gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
        else:
            gray = blur

        x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        absX = cv2.convertScaleAbs(x)
        grad = cv2.addWeighted(absX, 1, 0, 0, 0)

        _, threshold = cv2.threshold(grad, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

        element = cv2.getStructuringElement(cv2.MORPH_RECT, (morphW, morphH))
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, element)

        return threshold

    def sobelOperT(self, img, blursize, morphW, morphH):
        '''
            No different with sobelOper ? 
        '''
        blur = cv2.GaussianBlur(img, (blursize, blursize), 0, 0, cv2.BORDER_DEFAULT)

        if len(blur.shape) == 3:
            gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
        else:
            gray = blur

        x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, 3)
        absX = cv2.convertScaleAbs(x)
        grad = cv2.addWeighted(absX, 1, 0, 0, 0)

        _, threshold = cv2.threshold(grad, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

        element = cv2.getStructuringElement(cv2.MORPH_RECT, (morphW, morphH))
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, element)

        return threshold

    def verifySizes(self, mr):
        if mr[1][0] == 0 or mr[1][1] == 0:
            return False

        # life mode
        error = self.m_error
        aspect = self.m_aspect
        min = 34 * 8 * self.m_verifyMin
        max = 34 * 8 * self.m_verifyMax

        rmin = aspect - aspect * error
        rmax = aspect + aspect * error

        area = mr[1][0] * mr[1][1]  # height * width
        r = mr[1][0] / mr[1][1]

        if r < 1:
            r = 1 / r

        if (area < min or area > max) or (r < rmin or r > rmax):
            return False
        else:
            return True

    def calcSafeRect(self, roi, src):
        '''
            return [x, y, w, h]
        '''
        box = cv2.boxPoints(roi)
        x, y, w, h = cv2.boundingRect(box)

        src_h, src_w, _ = src.shape

        tl_x = x if x > 0 else 0
        tl_y = y if y > 0 else 0
        br_x = x + w - 1 if x + w - 1 < src_w else src_w - 1
        br_y = y + h - 1 if y + h - 1 < src_h else src_h - 1

        roi_w = br_x - tl_x
        roi_h = br_y - tl_y
        if roi_w <= 0 or roi_h <= 0:
            return [tl_x, tl_y, roi_w, roi_h], False

        return [tl_x, tl_y, roi_w, roi_h], True

    def rotation(self, in_img, rect_size, center, angle):
        '''
            rect_size: (h, w)
            rotation an image
        '''
        if len(in_img.shape) == 3:
            in_large = np.zeros((int(in_img.shape[0] * 1.5), int(in_img.shape[1] * 1.5), 3)).astype(in_img.dtype)
        else:
            in_large = np.zeros((int(in_img.shape[0] * 1.5), int(in_img.shape[1] * 1.5))).astype(in_img.dtype)

        x = int(max(in_large.shape[1] / 2 - center[0], 0))
        y = int(max(in_large.shape[0] / 2 - center[1], 0))

        width = int(min(in_img.shape[1], in_large.shape[1] - x))
        height = int(min(in_img.shape[0], in_large.shape[0] - y))

        if width != in_img.shape[1] and height != in_img.shape[0]:
            return in_img, False

        new_center = (in_large.shape[1] / 2, in_large.shape[0] / 2)

        rot_mat = cv2.getRotationMatrix2D(new_center, angle, 1)

        mat_rotated = cv2.warpAffine(in_large, rot_mat, (in_large.shape[1], in_large.shape[0]), cv2.INTER_CUBIC)

        img_crop = cv2.getRectSubPix(mat_rotated, (int(rect_size[0]), int(rect_size[0])), new_center)
        return img_crop, True

    def affine(self, in_mat, slope):
        height = in_mat.shape[0]
        width = in_mat.shape[1]
        xiff = abs(slope) * height
        if slope > 0:
            plTri = np.float32([[0, 0], [width - xiff - 1, 0], [xiff, height - 1]])
            dstTri = np.float32([[xiff / 2, 0], [width - 1 - xiff / 2, 0], [xiff / 2, height - 1]])
        else:
            plTri = np.float32([[xiff, 0], [width - 1, 0], [0, height - 1]])
            dstTri = np.float32([[xiff / 2, 0], [width - 1 - xiff / 2, 0], [xiff / 2, height - 1]])
        warp_mat = cv2.getAffineTransform(plTri, dstTri)

        if in_mat.shape[0] > 36 or in_mat.shape[1] > 136:
            affine_mat = cv2.warpAffine(in_mat, warp_mat, (int(height), int(width)), cv2.INTER_AREA)
        else:
            affine_mat = cv2.warpAffine(in_mat, warp_mat, (int(height), int(width)), cv2.INTER_CUBIC)
        return affine_mat

    def isdeflection(self, in_img, angle):
        comp_index = [in_img.shape[0] / 4, in_img.shape[0] / 2, in_img.shape[0] / 4 * 3]
        len = []
        for i in range(3):
            index = comp_index[i]
            j = 0
            value = 0
            while value == 0 and j < in_img.shape[1]:
                value = in_img[int(index), j]
                j += 1
            len.append(j)
        maxlen = max(len[2], len[0])
        minlen = min(len[2], len[0])
        difflen = abs(len[2] - len[0])
        PI = 3.1415926
        import math
        g = math.tan(angle * PI / 180)

        if (maxlen - len[1] > (in_img.shape[1] / 32)) or (len[1] - minlen > (in_img.shape[1] / 32)):
            slope_can_1 = (len[2] - len[0]) / comp_index[1]
            slope_can_2 = (len[1] - len[0]) / comp_index[0]
            slope_can_3 = (len[2] - len[1]) / comp_index[0]
            slope = slope_can_1 if abs(slope_can_1 - g) <= abs(slope_can_2 - g) else slope_can_2
            return slope, True
        else:
            slope = 0
        return slope, False

    def DeleteNotArea(self, in_img):
        input_gray = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
        w = in_img.shape[1]
        h = in_img.shape[0]
        tmp_mat = in_img[int(h * 0.1):int(h * 0.85), int(w * 0.15):int(w * 0.85)]

        plateType = getPlateType(tmp_mat, True)

        if plateType == 'BLUE':
            tmp = in_img[int(h * 0.1):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
            threadHoldV = ThresholdOtsu(tmp)
            _, img_threshold = cv2.threshold(input_gray, threadHoldV, 255, cv2.THRESH_BINARY)
        elif plateType == 'YELLOW':
            tmp = in_img[int(h * 0.1):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
            threadHoldV = ThresholdOtsu(tmp)
            _, img_threshold = cv2.threshold(input_gray, threadHoldV, 255, cv2.THRESH_BINARY_INV)
        else:
            _, img_threshold = cv2.threshold(input_gray, 10, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

        top, bottom = clearLiuDing(img_threshold, 0, img_threshold.shape[0] - 1)
        posLeft, posRight, flag = bFindLeftRightBound1(img_threshold)

        if flag:
            in_img = in_img[int(top):int(bottom), int(posLeft):int(w)]

    def plateColorLocate(self, src, cand):
        rects_blue = []
        rects_yellow = []
        src_b = self.colorSearch(src, Color.BLUE, rects_blue)
        self.deskew(src, src_b, rects_blue, cand)

        src_b = self.colorSearch(src, Color.YELLOW, rects_yellow)
        self.deskew(src, src_b, rects_yellow, cand)

    def colorSearch(self, src, color, out_rect):
        """

        :param src:
        :param color:
        :param out_rect: minAreaRect
        :return: binary
        """
        color_morph_width = 10
        color_morph_height = 2

        match_gray = colorMatch(src, color, False)

        _, src_threshold = cv2.threshold(match_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

        element = cv2.getStructuringElement(cv2.MORPH_RECT, (color_morph_width, color_morph_height))
        src_threshold = cv2.morphologyEx(src_threshold, cv2.MORPH_CLOSE, element)

        out = src_threshold.copy()

        _, contours, _ = cv2.findContours(src_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            mr = cv2.minAreaRect(cnt)
            if self.verifySizes(mr):
                out_rect.append(mr)

        return out
