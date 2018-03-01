# @Time    : 2018/2/9
# @Author  : fh
# @File    : find_last.py
# @Desc    :
"""
    find last checkpoint
"""
import os


def find_last(output_dir, prefix):
    # find the last checkpoint
    dir_names = next(os.walk(output_dir))[1]
    dir_names = filter(lambda f: f.startswith(prefix), dir_names)
    dir_names = sorted(dir_names)
    assert dir_names, 'There is no trained model. Please download or train.'
    return dir_names[-1]
