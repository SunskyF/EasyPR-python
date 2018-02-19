# @Time    : 2018/2/9
# @Author  : fh
# @File    : preprocess_chars.py
# @Desc    :
"""
    pre-process easypr data
"""
import os
from six.moves import cPickle as pickle
import random

DATA_DIR = '../data/easypr_train_data/'


def generate_label(cls_dir, labels):
    total_list = []

    cnt = 0
    for label in labels:
        for name in os.listdir(os.path.join(DATA_DIR, cls_dir, label)):
            record = {'name': name, 'label': cnt, 'subdir': label}
            total_list.append(record)
        cnt += 1
    random.shuffle(total_list)
    train_size = int(0.7 * len(total_list))
    print(train_size, len(total_list))

    with open(os.path.join(DATA_DIR, cls_dir, 'train.pickle'), 'wb') as f:
        pickle.dump(total_list[:train_size], f, 2)

    with open(os.path.join(DATA_DIR, cls_dir, 'val.pickle'), 'wb') as f:
        pickle.dump(total_list[train_size:], f, 2)


if __name__ == '__main__':
    chars_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                    'W', 'X', 'Y', 'Z', 'zh_cuan', 'zh_e', 'zh_gan',
                    'zh_gan1', 'zh_gui', 'zh_gui1', 'zh_hei', 'zh_hu',
                    'zh_ji', 'zh_jin', 'zh_jing', 'zh_jl', 'zh_liao',
                    'zh_lu', 'zh_meng', 'zh_min', 'zh_ning', 'zh_qing',
                    'zh_qiong', 'zh_shan', 'zh_su', 'zh_sx', 'zh_wan',
                    'zh_xiang', 'zh_xin', 'zh_yu', 'zh_yu1', 'zh_yue',
                    'zh_yun', 'zh_zang', 'zh_zhe']
    generate_label('chars', chars_labels)

    is_labels = ['no', 'has']
    generate_label('whether_car', is_labels)
