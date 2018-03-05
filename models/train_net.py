# @Time    : 2018/2/9
# @Author  : fh
# @File    : train_net.py
# @Desc    :
"""
    Train entry
"""
# easypr train
from easypr.dataset import DataSet
from easypr.net.lenet import Lenet
from easypr.net.judgenet import Judgenet
from easypr.cnn_train import Train

# mask-rcnn train
from mrcnn.plate import PlateConfig, PlateDataset
import mrcnn.model as modellib

# multi-label train
from multilabel.train import MultiLabelSolver

import argparse
import os
import sys

sys.path.append('../lib')
from config import cfg, cfg_from_file


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Func test ULPR')
    parser.add_argument('--data_dir', dest='data_dir', required=True,
                        help='data dir', default=None, type=str)
    parser.add_argument('--output_dir', dest='output_dir', required=True,
                        help='output dir', default=None, type=str)
    parser.add_argument('--batch_size', dest='batch_size', required=False,
                        help='batch size', default=32, type=int)
    parser.add_argument('--lr', dest='lr', required=False,
                        help='learning rate', default=0.01, type=float)
    parser.add_argument('--epoch', dest='epoch', required=False,
                        help='epoch', default=10, type=int)
    parser.add_argument('--task', dest='task', required=True,
                        help='the task to be trained, (char, judge, mrcnn, multilabel)', default=None, type=str)
    parser.add_argument('--net', dest='net', required=False,
                        help='the net to be trained, only in task (multilabel: vgg16)',
                        default=None, type=str)
    parser.add_argument('--gpu', dest='gpu', required=False,
                        help='which gpu to use', default='0', type=str)
    parser.add_argument('--weights', dest='pretrained_model', required=False,
                        help='use pretrained model', default=None, type=str)
    parser.add_argument('--cfg', dest='cfg', required=False,
                        help='the config file', default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.cfg:
        cfg_from_file(args.cfg)

    if args.task in ['char', 'judge']:
        dataset_params = {
            'batch_size': args.batch_size,
            'path': args.data_dir,
            'thread_num': 3
        }

        if args.net == 'char':
            model = Lenet()
            dataset_params['gray'] = True
        elif args.net == 'judge':
            model = Judgenet()

        dataset_train = DataSet(dataset_params, 'train')
        dataset_params['batch_size'] = 100
        dataset_val = DataSet(dataset_params, 'val')

        params = {
            'lr': args.lr,
            'number_epoch': args.epoch,
            'epoch_length': dataset_train.record_number,
            'log_dir': args.output_dir
        }

        model.compile()
        train = Train(params)
        train.compile(model)
        train.train(dataset_train, dataset_val)
    elif args.task in ['mrcnn']:
        config = PlateConfig()
        config.display()

        # Training dataset
        print('Reading data...')
        dataset_train = PlateDataset()
        dataset_train.load_plate(args.data_dir, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = PlateDataset()
        dataset_val.load_plate(args.data_dir, "val")
        dataset_val.prepare()

        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.output_dir)
        model_path = args.pretrained_model
        exclude = ["mrcnn_class_logits", "mrcnn_bbox_fc",
                   "mrcnn_bbox", "mrcnn_mask"]
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True, exclude=exclude)
        # This training schedule is an example. Update to fit your needs.

        # Training - Stage 1
        # Adjust epochs and layers as needed
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=10,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Training Resnet layer 4+")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=20,
                    layers='4+')

        # Training - Stage 3
        # Finetune layers from ResNet stage 3 and up
        print("Training Resnet layer 3+")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 100,
                    epochs=30,
                    layers='all')
    elif args.task in ['multilabel']:
        assert args.net is not None, "Please specify the backbone net"
        solver = MultiLabelSolver(args.net, cfg)
        solver.train()
    else:
        raise NotImplementedError('The task is not supported.')
