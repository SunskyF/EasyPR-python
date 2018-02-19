import os
import random
import cv2
import numpy as np
from queue import Queue
from threading import Thread
from six.moves import cPickle as pickle


class DataSet(object):
    def __init__(self, dataset_params, phase):

        # process params
        self.data_path = str(dataset_params['path'])
        self.label_path = os.path.join(self.data_path, phase + '.pickle')
        self.batch_size = int(dataset_params['batch_size'])
        self.thread_num = int(dataset_params['thread_num'])
        self.gray = False

        if 'gray' in dataset_params:
            self.gray = dataset_params['gray']

        # record and image_label queue
        self.record_queue = Queue(maxsize=10000)
        self.image_label_queue = Queue(maxsize=512)

        with open(self.label_path, 'rb') as f:
            result = pickle.load(f)

        self.record_list = result  # {'name', 'label', 'subdir'}
        self.record_point = 0
        self.record_number = len(self.record_list)
        if self.batch_size == -1:
            self.batch_size = self.record_number

        self.num_batch_per_epoch = int(self.record_number / self.batch_size)

        t_record_producer = Thread(target=self.record_producer)
        t_record_producer.daemon = True
        t_record_producer.start()

        for i in range(self.thread_num):
            t = Thread(target=self.record_customer)
            t.daemon = True
            t.start()

    def record_producer(self):
        """record_queue's processor
        """
        while True:
            if self.record_point % self.record_number == 0:
                random.shuffle(self.record_list)
                self.record_point = 0

            self.record_queue.put([os.path.join(self.data_path, self.record_list[self.record_point]['subdir'],
                                                self.record_list[self.record_point]['name']),
                                   self.record_list[self.record_point]['label']])
            self.record_point += 1

    def record_process(self, record, gray=False):
        """record process
        Args: record
        Returns:
          image: 3-D ndarray
          labels: 2-D list
        """
        if gray:
            image = cv2.imdecode(np.fromfile(record[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)[..., None]
        else:
            image = cv2.imdecode(np.fromfile(record[0], dtype=np.uint8), cv2.IMREAD_COLOR)
        return [image, record[1]]

    def record_customer(self):
        """record queue's customer 
        """
        while True:
            item = self.record_queue.get()
            out = self.record_process(item, self.gray)
            self.image_label_queue.put(out)

    def batch(self):
        """get batch
        Returns:
          images: 4-D ndarray [batch_size, height, width, 1]
          labels: 1-D ndarray [batch_size, ]
        """
        images = []
        labels = []

        for i in range(self.batch_size):
            image, label = self.image_label_queue.get()
            images.append(image)
            labels.append(label)

        images = np.asarray(images, dtype=np.float32)
        images = images / 255 * 2 - 1

        return images, labels
