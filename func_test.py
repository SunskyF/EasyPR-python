# @Time    : 2018/2/7
# @Author  : fh
# @File    : func_test.py
# @Desc    :
"""
    Functional test
"""

from lib.config import cfg_from_file
from plate import *

import sys
import argparse


test_menu = (
    '-' * 8 + '\n' +
    '功能测试:\n' +
    '1. test plate_detect(车牌检测);\n' +
    '2. test chars_recognize(字符识别);\n' +
    '3. test plate_recognize(车牌识别);\n' +
    '-' * 8
)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Func test ULPR')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='must have a config file', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def main():
    while True:
        print(test_menu)
        select = input()
        test_op[select]()

test_op = {
    '1': test_plate_detect,
    '2': test_chars_recognize,
    '3': test_plate_recognize,
}

if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    main()
