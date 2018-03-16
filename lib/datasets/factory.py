# @Time    : 2018/3/11
# @Author  : fh
# @File    : factory.py
# @Desc    :
"""
    factory to get different dataset
"""

from .plate_char import PlateCharDataset

__sets = {}

# plate char dataset
__sets['plate_char'] = PlateCharDataset

def get_dataset(name, split, cfg):
    """
        get a dataset by name
    :param name: dataset name
    :param split: train or val
    :return: dataset class
    """
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    if split not in ['train', 'val']:
        raise KeyError('Unknown split: {}'.format(split))
    return __sets[name](split, cfg)
