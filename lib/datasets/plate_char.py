# @Time    : 2018/3/10
# @Author  : fh
# @File    : plate_char.py
# @Desc    :
"""
    dataset to read plate char
"""
import os
import numpy as np
import cv2

from .dataset import Dataset


class PlateCharDataset(Dataset):
    def load_data(self, data_dir):
        """Load small plate char dataset.
        dataset_dir: The root directory of the small plate dataset.
        subset: What to load (train, val)
        return_plate: If True, returns the plate object.
        """
        # Path
        image_dir = os.path.join(data_dir, "JPGImages")

        # Create plate object
        imageset_file = os.path.join(data_dir, self.split + '.txt')
        with open(imageset_file, 'r', encoding='utf-8') as f:
            image_index = []
            image_anno = []
            for line in f.readlines():
                line = line.strip().split()
                image_index.append(line[0])
                image_anno.append(line[1])

        # Add classes
        for i in range(1, len(self.cfg.CLASSES) + 1):
            self.add_class(1, self.cfg.CLASSES[i - 1])

        for i, idx in enumerate(image_index):
            image_path = os.path.join(image_dir, idx + '.jpg')
            assert os.path.exists(image_path), "{} is not exist".format(image_path)
            self.add_image(
                image_id=i,
                path=image_path,
                annotation=image_anno[i],
                origin_name=idx,
            )
        self.prepare()

    def load_annotation(self, image_id):
        annotation = self.image_info[image_id]['annotation']
        assert len(annotation) <= 7, "Unsupport Plate Length"
        label = np.zeros(7)
        for idx, char in enumerate(annotation):
            ind = self._class_to_ind[char]
            label[idx] = ind
        return label

    def load_gt(self, image_id, augment=False):
        image = self.load_image(image_id)
        image = cv2.resize(image, (self.image_resize_width, self.image_resize_height))
        label = self.load_annotation(image_id)
        return image, label
