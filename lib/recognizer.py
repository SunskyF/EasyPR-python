# @Time    : 2018/2/7
# @Author  : fh
# @File    : recognizer.py
# @Desc    :
"""
    The recognizer wrapper of ULPR. It will recognize the plate license of the plate roi.
    If you want to add a method, add the name in methods and the corresponding function at the end of the file.
"""

from lib.config import cfg
from lib.utils.find_last import find_last

# lib methods
from lib.easypr.chars_recognize import CharsRecognize

_methods = [
    'easypr',  # 0
]


def recognize(src):
    """
        Recognizer factory
    :param src: plate roi
    :return:
    """
    method = cfg.RECOGNIZER.METHOD
    return eval('_recognize_' + _methods[method])(src)


# method 0
def _recognize_easypr(src):
    cr = CharsRecognize()
    dir_name = find_last(cfg.OUTPUT_DIR, 'chars')
    return cr.charsRecognize(src, str(cfg.OUTPUT_DIR / dir_name / 'models'))
