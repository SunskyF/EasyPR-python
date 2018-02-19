#!/usr/bin/env bash
python models/train_net.py --data_dir data/easypr_train_data/chars --output_dir output/chars --batch_size 32 --lr 0.01  --net char --epoch 10