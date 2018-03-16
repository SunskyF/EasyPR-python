#!/usr/bin/env bash
python models/train_net.py --data_dir data/plate_chars --task multilabel --net vgg16 --weights data/pretrained/vgg_16.ckpt --cfg cfgs/train/multilabel.yml