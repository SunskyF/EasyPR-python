import argparse
import os
import time
import matplotlib.pyplot as plt

import cv2
import numpy as np

from lib.config import cfg_from_file, cfg
from lib.detector import detect as plate_detect
from lib.recognizer import recognize as chars_recognize
from lib.utils.align import align

plt.rcParams['font.sans-serif'] = ['SimHei']  # display chinese title in plt
plt.rcParams['axes.unicode_minus'] = False  # display minus normally


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='ULPR demo')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='config file', default='cfgs/easypr.yml', type=str)
    parser.add_argument('--data', dest='data_dir',
                        help='test image dir', default='data/general_test', type=str)

    args = parser.parse_args()
    return args


def accuracy_test(data_dir):
    print("Begin to test accuracy")
    count = [0, 0]  # total images, correct images
    not_recognized_names = []
    image_names = os.listdir(data_dir)
    starttime = time.time()

    for image_name in image_names:
        print('-' * 8)
        count[0] += 1
        label = image_name.split('.')[0]
        # read Chinese plate
        src = cv2.imdecode(np.fromfile(os.path.join(data_dir, image_name), dtype=np.uint8), cv2.IMREAD_COLOR)
        print("Label: ", label)
        time0 = time.time()
        results = plate_detect(src)
        for res in results:
            vis_image = align(src, res)
            rec_res = chars_recognize(vis_image)
            print("Chars Recognise: ", rec_res)
            if label == rec_res:
                count[1] += 1
                break
            else:
                if cfg.DEBUG:
                    plt.title(rec_res)
                    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
                    plt.show()
        print("time: {}s".format(time.time() - time0))
        print('-' * 8)

    endtime = time.time()
    print("Accuracy test end!")
    print("Summary:")
    print("Total time: {:.2f}s, Average time: {:.2f}s".format(endtime - starttime, (endtime - starttime) / count[0]))
    print("Accuracy: {:.2f}%({})".format(count[1] / count[0] * 100, count[0]))
    print("Not recognize: ")
    for pic in not_recognized_names:
        print(pic)

if __name__ == "__main__":
    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    cfg.VIS = False
    accuracy_test(args.data_dir)

