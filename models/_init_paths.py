# @Time    : 2018/3/7
# @Author  : fh
# @File    : _init_paths.py.py
# @Desc    :
"""
    Add init paths
"""
import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.append(path)


this_dir = osp.dirname(__file__)

lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)
