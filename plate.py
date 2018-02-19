import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from lib.config import cfg
from lib.detector import detect as plate_detect
from lib.recognizer import recognize as chars_recognize
from lib.utils.align import align

plt.rcParams['font.sans-serif'] = ['SimHei']  # display chinese title in plt
plt.rcParams['axes.unicode_minus'] = False  # display minus normally


def test_plate_detect():
    print("Testing Plate Detect")

    file = cfg.DATA_DIR / 'demo' / 'test.jpg'

    src = cv2.imread(str(file))
    if cfg.VIS and 0:
        plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        plt.show()
    results = plate_detect(src)
    if cfg.VIS:
        for res in results:
            print("Plate position: \n", res)
            fig = plt.figure(figsize=(10, 10))
            ax1 = fig.add_subplot(211)
            ax1.imshow(src)
            ax1.add_patch(patches.Polygon(res))
            ax2 = fig.add_subplot(212)
            vis_image = align(src, res)
            ax2.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            plt.show()


def test_chars_recognize():
    print("Testing Chars Recognize")

    file = cfg.DATA_DIR / 'demo' / 'chars_recognize.jpg'
    assert file.exists()
    src = cv2.imread(str(file))
    if cfg.VIS:
        plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        plt.show()

    rec_res = chars_recognize(src)
    print("Chars Recognize: {} ({})".format(rec_res, 'CORRECT' if rec_res == '沪AGH092' else 'WRONG'))


def test_plate_recognize():
    print("Testing Plate Recognize")

    file = cfg.DATA_DIR / 'demo' / 'test.jpg'

    src = cv2.imread(str(file))

    results = plate_detect(src)

    for res in results:
        print("Plate position: \n", res)
        vis_image = align(src, res)
        rec_res = chars_recognize(vis_image)
        print("Chars Recognize: ", rec_res)
        if cfg.VIS:
            plt.title(rec_res)
            plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            plt.show()
