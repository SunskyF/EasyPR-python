# @Time    : 2018/2/9
# @Author  : fh
# @File    : sort_points.py
# @Desc    :
"""
    Sort points of quadrilateral
"""
import numpy as np


def sort_points(points):
    """
        sort points
         -> x
        \
        y
           0             3
            \-----------\
            \           \
            \           \
            \-----------\
           1             2
    :param points: [4, 2] (x, y)
    :return: sorted_points: [4, 2] (x, y)
    """
    x = points[:, 0]
    x_sort = np.argsort(x)
    sorted_by_x = points[x_sort]
    sorted_approx = []

    p0 = sorted_by_x[0]
    p1 = sorted_by_x[1]
    p2 = sorted_by_x[2]
    l0 = np.linalg.norm(p0 - p1)
    l1 = np.linalg.norm(p0 - p2)
    l2 = np.linalg.norm(p1 - p2)

    min_index = np.argmin([l0, l1, l2])
    rest_points = [sorted_by_x[3]]
    if min_index == 0:
        if p0[1] < p1[1]:
            sorted_approx.append(p0)
            sorted_approx.append(p1)
        else:
            sorted_approx.append(p1)
            sorted_approx.append(p0)
        rest_points.append(p2)
    elif min_index == 1:
        if p0[1] < p2[1]:
            sorted_approx.append(p0)
            sorted_approx.append(p2)
        else:
            sorted_approx.append(p2)
            sorted_approx.append(p0)
        rest_points.append(p1)
    elif min_index == 2:
        if p1[1] < p2[1]:
            sorted_approx.append(p1)
            sorted_approx.append(p2)
        else:
            sorted_approx.append(p2)
            sorted_approx.append(p1)
        rest_points.append(p0)
    if rest_points[0][1] > rest_points[1][1]:
        sorted_approx.append(rest_points[0])
        sorted_approx.append(rest_points[1])
    else:
        sorted_approx.append(rest_points[1])
        sorted_approx.append(rest_points[0])

    return np.array(sorted_approx)


if __name__ == '__main__':
    test_points = np.array([[871., 807.],
                            [871., 774.],
                            [1012., 774.],
                            [1012., 807.]])
    print("Before sort: ", test_points)
    print("Sorted: ", sort_points(test_points))
