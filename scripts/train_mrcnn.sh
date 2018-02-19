#!/usr/bin/env bash
python models/train_net.py --data_dir data/plate --output_dir output --net mrcnn --weights data/pretrained/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5