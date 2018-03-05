#!/usr/bin/env bash
python models/train_net.py --data_dir data/easypr_train_data/whether_car --output_dir output/whether_car --batch_size 32 --lr 0.01 --task judge --epoch 10