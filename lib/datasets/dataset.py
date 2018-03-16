# @Time    : 2018/3/10
# @Author  : fh
# @File    : dataset.py
# @Desc    :
"""
    In the future, it will be the base dataset class
    Reference from https://github.com/matterport/Mask_RCNN
"""
import cv2
import numpy as np


class Dataset:
    """
    The base class for dataset classes.
    """
    def __init__(self, split, cfg):
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"id": 0, "name": "BG"}]
        self.source_class_ids = {}
        self.split = split
        self.is_train = split == 'train'
        self.cfg = cfg
        if self.is_train:
            self.image_resize_width = self.cfg.TRAIN.IMAGE_WIDTH
            self.image_resize_height = self.cfg.TRAIN.IMAGE_HEIGHT
        else:
            self.image_resize_width = self.cfg.TEST.IMAGE_WIDTH
            self.image_resize_height = self.cfg.TEST.IMAGE_HEIGHT

    def add_class(self, class_id, class_name):
        # Add the class
        self.class_info.append({
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def prepare(self):
        """Prepares the Dataset class for use.
        """
        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [c["name"] for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)
        self._class_to_ind = dict(list(zip(self.class_names, self.class_ids)))

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id, extract_mean=True):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = cv2.imread(self.image_info[image_id]['path']).astype(np.float32, copy=False)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if extract_mean:
            image -= self.cfg.PIXEL_MEAN
        return image

    def next_batch(self, shuffle=True):
        """
            If the dataset is different, you may need override this function.
            In the ordinary times, you just need override load_gt.
        :param shuffle: shuffle the index
        :param augment: whether use some augment ways
        :return: [image, label]
        """
        # TODO: prefetch
        batch_size = self.cfg.SOLVER.BATCH_SIZE
        assert len(self._image_ids) >= batch_size, "Images less than batch size"

        while True:
            if shuffle:
                np.random.shuffle(self._image_ids)
            images = []
            labels = []
            for i in self._image_ids:
                image, label = self.load_gt(i)
                images.append(image)
                labels.append(label)
                if len(images) == batch_size:
                    yield np.array(images), np.array(labels)
                    images = []
                    labels = []


    # Need to override
    def load_data(self, data_dir):
        raise NotImplementedError

    def load_annotation(self, image_id):
        raise NotImplementedError

    def load_gt(self, image_id, augment=False):
        """
            use load_data and load_annotation to generate gt images and labels
            used in next_batch
        :return:
        """
        raise NotImplementedError

