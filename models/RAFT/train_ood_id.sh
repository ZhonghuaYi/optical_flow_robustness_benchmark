#!/bin/bash
mkdir -p checkpoints

# train OOD model
python -u train.py --name raft-ofb-ood --stage ood --validation kitti_c --restore_ckpt models/raft-things.pth --gpus 0 --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.00001 --gamma=0.85

# train ID model
CUDA_VISIBLE_DEVICES=6 python -u train.py --name raft-ofb-id  --stage id --validation kitti_c --restore_ckpt models/raft-ofb-ood.pth --gpus 0 --num_steps 50000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85