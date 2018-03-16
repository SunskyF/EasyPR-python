# @Time    : 2018/2/9
# @Author  : fh
# @File    : train_net.py
# @Desc    :
"""
    Train entry
"""
import _init_paths
from datasets.factory import get_dataset

import argparse
import os
import sys

from config import cfg, cfg_from_file


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Func test ULPR')
    parser.add_argument('--data_dir', dest='data_dir', required=True,
                        help='data dir', default=None, type=str)
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
        # easypr train
        from easypr.dataset import DataSet
        from easypr.net.lenet import Lenet
        from easypr.net.judgenet import Judgenet
        from easypr.cnn_train import Train

        dataset_params = {
            'batch_size': cfg.SOLVER.BATCH_SIZE,
            'path': args.data_dir,
            'thread_num': 3
        }
        if args.task == 'char':
            model = Lenet()
            dataset_params['gray'] = True
            model_prefix = "chars"
        elif args.task == 'judge':
            model = Judgenet()
            model_prefix = "whether_car"

        dataset_train = DataSet(dataset_params, 'train')
        dataset_params['batch_size'] = 100
        dataset_val = DataSet(dataset_params, 'val')

        params = {
            'lr': cfg.SOLVER.LEARNING_RATE,
            'number_epoch': cfg.SOLVER.EPOCH,
            'epoch_length': dataset_train.record_number,
            'log_dir': cfg.OUTPUT_DIR / model_prefix
        }

        model.compile()
        train = Train(params)
        train.compile(model)
        train.train(dataset_train, dataset_val)
    elif args.task in ['mrcnn']:
        # mask-rcnn train
        from mrcnn.plate import PlateConfig, PlateDataset
        import mrcnn.model as modellib

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
        # multi-label train
        from multilabel.train import MultiLabelSolver

        assert args.net is not None, "Please specify the backbone net"
        # Training dataset
        print('Reading data...')
        dataset_train = get_dataset('plate_char', 'train', cfg)
        dataset_train.load_data(args.data_dir)

        # Validation dataset
        dataset_val = get_dataset('plate_char', 'val', cfg)
        dataset_val.load_data(args.data_dir)

        # Train model
        solver = MultiLabelSolver(args.net, cfg, dataset_train, dataset_val, pretrained_model=args.pretrained_model)
        solver.train()
    else:
        raise NotImplementedError('The task is not supported.')
