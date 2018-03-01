import matplotlib.pyplot as plt
import numpy as np
import cv2
from enum import Enum

Color = Enum('Color', ('BLUE', 'YELLOW', 'WHITE', 'UNKNOW'))

def clearLiuDingOnly(img):
    x = 7#default 7 char
    jump = np.zeros((img.shape[0]))

    for i in range(img.shape[0]):
        jumpCnt = 0
        whiteCnt = 0
        for j in range(img.shape[1] - 1):
            if img[i][j] != img[i][j + 1]:
                jumpCnt += 1
            if img[i][j] == 255:
                whiteCnt += 1
        jump[i] = jumpCnt

    for i in range(img.shape[0]):
        if jump[i] <= x:
            for j in range(img.shape[1]):
                img[i][j] = 0


def getPlateType(src, adaptive_minsv):
    blue_per, flag = plateColorJudge(src, 'BLUE', adaptive_minsv)
    if flag:
        return Color.BLUE
    yellow_per, flag = plateColorJudge(src, 'YELLOW', adaptive_minsv)
    if flag:
        return Color.YELLOW
    white_per, flag = plateColorJudge(src, 'WHITE', adaptive_minsv)
    if flag:
        return Color.WHITE
    tmp = np.array([blue_per, yellow_per, white_per])

    tmp_color = [Color.BLUE, Color.YELLOW, Color.WHITE]
    return tmp_color[tmp.argmax()]

def clearLiuDingChar(img):
    x = 7
    jump = np.zeros((img.shape[0]))
    fjump = []
    whiteCnt = 0

    for i in range(img.shape[0]):
        jumpCnt = 0

        for j in range(img.shape[1] - 1):
            if img[i][j] != img[i][j + 1]:
                jumpCnt += 1
            if img[i][j] == 255:
                whiteCnt += 1
        jump[i] = jumpCnt

    icount = 0
    for i in range(img.shape[0]):
        fjump.append(jump[i])
        if jump[i] >= 16 and jump[i] <= 45:
            icount += 1

    if icount / img.shape[0] <= 0.4:
        return False

    if whiteCnt / (img.shape[0] * img.shape[1]) < 0.15 or whiteCnt / (img.shape[0] * img.shape[1]) > 0.5:
        return False

    for i in range(img.shape[0]):
        if jump[i] <= x:
            for j in range(img.shape[1]):
                img[i, j] = 0

    return True

def clearLiuDing(mask, top, bottom):
    x = 7
    for i in range(mask.shape[0] // 2):
        jumpCnt = 0
        whiteCnt = 0
        for j in range(mask.shape[1] - 1):
            if mask[i][j] != mask[i][j + 1]:
                jumpCnt += 1
            if mask[i][j] == 255:
                whiteCnt += 1
        if (jumpCnt < x and whiteCnt / mask.shape[1] > 0.15) or whiteCnt < 4:
            top = i
    top -= 1
    top = max(top, 0)
    for i in range(mask.shape[0] - 1, mask.shape[0] // 2 - 1, -1):
        jumpCnt = 0
        whiteCnt = 0
        for j in range(mask.shape[1] - 1):
            if mask[i][j] != mask[i][j + 1]:
                jumpCnt += 1
            if mask[i][j] == 255:
                whiteCnt += 1
        if (jumpCnt < x and whiteCnt / mask.shape[1] > 0.15) or whiteCnt < 4:
            bottom = i
    bottom += 1
    bottom = min(bottom, mask.shape[0] - 1)
    if (top >= bottom):
        top = 0
        bottom = mask.shape[0] - 1
    return top, bottom


def ThresholdOtsu(mat):
    height = mat.shape[0]
    width = mat.shape[1]

    histogram = np.zeros((256))
    for i in range(height):
        for j in range(width):
            histogram[mat[i, j]] += 1
    histogram /= height * width
    avgvalue = np.sum(np.arange(256) * histogram)

    w = 0
    u = 0
    max_var = 0
    thresholdV = 0
    for i in range(256):
        w += histogram[i]
        u += i * histogram[i]

        t = avgvalue * w - u
        variance = t * t / (w * (1 - w) + 0.0001)
        if variance > max_var:
            max_var = variance
            thresholdV = i
    return thresholdV


def bFindLeftRightBound1(bound_threshold):
    span = bound_threshold.shape[0] * 0.2

    for i in range(0, bound_threshold.shape[1] - int(span) - 1, 3):
        whiteCnt = 0
        for k in range(bound_threshold.shape[0]):
            for l in range(i, i + int(span)):
                if bound_threshold[k, l] == 255:
                    whiteCnt += 1

        if (whiteCnt / (span * bound_threshold.shape[0])) > 0.15:
            posLeft = i
            break
    span = bound_threshold.shape[0] * 0.2
    posLeft = 0
    posRight = 0
    for i in range(bound_threshold.shape[1] - 1, int(span), -2):
        whiteCnt = 0
        for k in range(bound_threshold.shape[0]):
            for l in range(i, i - int(span), -1):
                if bound_threshold[k, l] == 255:
                    whiteCnt += 1

        if (whiteCnt / (span * bound_threshold.shape[0])) > 0.06:
            posRight = i
            if posRight + 5 < bound_threshold.shape[1]:
                posRight += 5
            else:
                posRight = bound_threshold.shape[1] - 1
            break
    if posLeft < posRight:
        return posLeft, posRight, True
    else:
        return posLeft, posRight, False


def plateColorJudge(src, r, adaptive_minsv):
    thresh = 0.45
    src_gray = colorMatch(src, r, adaptive_minsv)
    percent = cv2.countNonZero(src_gray) / (src_gray.shape[0] * src_gray.shape[1])

    if percent > thresh:
        return percent, True
    else:
        return percent, False


def colorMatch(src, r, adaptive_minsv):
    max_sv = 255
    minref_sv = 64
    minabs_sv = 95

    blue_region = [100, 140]
    yellow_region = [15, 40]
    white_region = [0, 30]

    src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    hsv_split = cv2.split(src_hsv)
    hsv_split[2] = cv2.equalizeHist(hsv_split[2])
    src_hsv = cv2.merge(hsv_split)

    if r == Color.BLUE:
        min_max_h = blue_region
    elif r == Color.YELLOW:
        min_max_h = yellow_region
    elif r == Color.WHITE:
        min_max_h = white_region
    else:
        min_max_h = [0, 0]

    diff_h = (min_max_h[1] - min_max_h[0]) / 2
    avg_h = min_max_h[0] + diff_h

    s_all = 0
    v_all = 0
    count = 0
    for i in range(src_hsv.shape[0]):
        for j in range(src_hsv.shape[1]):
            H = src_hsv[i, j, 0]
            S = src_hsv[i, j, 1]
            V = src_hsv[i, j, 2]

            s_all += S
            v_all += V
            count += 1
            colorMatched = False

            if H > min_max_h[0] and H < min_max_h[1]:
                if H > avg_h:
                    Hdiff = H - avg_h
                else:
                    Hdiff = avg_h - H

                Hdiff_p = Hdiff / diff_h

                min_sv = 0

                if adaptive_minsv:
                    min_sv = minref_sv - minref_sv / 2 * (1 - Hdiff_p)
                else:
                    min_sv = minabs_sv

                if (S > min_sv and S < max_sv) and (V > min_sv and V < max_sv):
                    colorMatched = True
            if colorMatched:
                src_hsv[i, j, 0] = 0
                src_hsv[i, j, 1] = 0
                src_hsv[i, j, 2] = 255
            else:
                src_hsv[i, j, 0] = 0
                src_hsv[i, j, 1] = 0
                src_hsv[i, j, 2] = 0

    hsvsplit_done = cv2.split(src_hsv)
    src_gray = hsvsplit_done[2]

    return src_gray


def bFindLeftRightBound(bound_threshold):
    span = bound_threshold.shape[0] * 0.2

    posLeft = 0
    posRight = 0
    for i in range(0, bound_threshold.shape[1] - int(span) - 1, 2):
        whiteCnt = 0
        for k in range(bound_threshold.shape[0]):
            for l in range(i, i+int(span)):
                if bound_threshold[k, l] == 255:
                    whiteCnt += 1

        if (whiteCnt / (span * bound_threshold.shape[0])) > 0.36:
            posLeft = i
            break
    span = bound_threshold.shape[0] * 0.2

    for i in range(bound_threshold.shape[1] - 1, int(span), -2):
        whiteCnt = 0
        for k in range(bound_threshold.shape[0]):
            for l in range(i, i - int(span), -1):
                if bound_threshold[k, l] == 255:
                    whiteCnt += 1

        if (whiteCnt / (span * bound_threshold.shape[0])) > 0.26:
            posRight = i
            break

    if posLeft < posRight:
        return posLeft, posRight, True
    else:
        return posLeft, posRight, False
