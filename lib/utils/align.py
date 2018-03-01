# @Time    : 2018/2/9
# @Author  : fh
# @File    : align.py
# @Desc    :
"""
    align the plate roi
"""
import numpy as np
import cv2


def align(image, points):
    """
    :param image:
    :param points:
    :return: aligned image
    """
    # alignment
    origin_point = np.require(np.array(points).reshape((4, 2)), dtype=np.single)
    height = int(max(np.linalg.norm(origin_point[0] - origin_point[1]), np.linalg.norm(origin_point[2] - origin_point[3])))
    width = int(max(np.linalg.norm(origin_point[0] - origin_point[3]), np.linalg.norm(origin_point[1] - origin_point[2])))

    target_point = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
    map_matrix = cv2.getPerspectiveTransform(origin_point, target_point)
    cols = width + 1
    rows = height + 1
    color = cv2.warpPerspective(image, map_matrix, (cols, rows))
    return color
