import os
import numpy as np
import cv2

from .config import Config
from . import utils


############################################################
#  Configurations
############################################################

class PlateConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "mrcnn"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384

    USE_MINI_MASK = False
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # COCO has 80 classes

    MAX_GT_INSTANCES = 3


class PlateInferenceConfig(PlateConfig):
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_DIM = 512


############################################################
#  Dataset
############################################################

class PlateDataset(utils.Dataset):
    def load_plate(self, dataset_dir):
        """Load small plate dataset.
        dataset_dir: The root directory of the small plate dataset.
        subset: What to load (train, val)
        return_plate: If True, returns the plate object.
        """
        # Path
        image_dir = os.path.join(dataset_dir, "JPGImages")
        anno_dir = os.path.join(dataset_dir, "Annotations")

        # Create plate object
        imageset_file = os.path.join(dataset_dir, subset + '.txt')
        with open(imageset_file, 'r', encoding=' gbk') as f:
            image_index = [x.strip().split('.')[0] for x in f.readlines()]
        # Add classes
        self.add_class("plates", 1, "plate")

        for i, idx in enumerate(image_index):
            image_path = os.path.join(image_dir, idx + '.jpg')
            assert os.path.exists(image_path), "{} is not exist".format(image_path)
            self.add_image(
                "plates", image_id=i,
                path=image_path,
                annotations=os.path.join(anno_dir, idx + '.txt'),
                origin_name=idx,
            )

    def load_mask(self, image_id, image_shape):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        instance_masks = []
        class_ids = []
        anno_path = self.image_info[image_id]['annotations']
        with open(anno_path, 'r', encoding='utf-8') as f:
            annotations = [x.strip() for x in f.readlines()]

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            m = self.annToMask(annotation, image_shape[0],
                               image_shape[1])

            instance_masks.append(m)
            class_ids.append(1)
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(self.__class__).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the plate dataset"""
        info = self.image_info[image_id]
        if info["source"] == "plates":
            return info[image_id]['origin_name']
        else:
            super(self.__class__).image_reference(self, image_id)

    def annToMask(self, ann, height, width):
        """
        Convert annotation to binary mask.
        :return: binary mask (numpy 2D array)
        """
        points = ann.split()[1:]
        points = list(map(int, points))
        assert len(points) == 8, 'points length wrong'
        points = np.array(points).reshape(-1, 2)
        mask = np.zeros((height, width))
        cv2.fillConvexPoly(mask, points, 1)
        return mask
