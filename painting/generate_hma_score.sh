#!/bin/bash

CURPATH=$(cd "$(dirname "$0")"; pwd)
cd $CURPATH

cd hma

mkdir -p assets/seg_weights

pip install gdown
gdown --id 1lse0Mqf7ny5qqV99nGQ3ccXTKJ6kNGoH
mv cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth ./assets/seg_weights
gdown --id 1DTY2qu8N3vD-Yw7OAEEQORqzpotxSMZk 
mv hrnetv2_w48_imagenet_pretrained.pth ./assets/seg_weights

python val.py --eval_folder ../../detector/data/kitti/training/ --result_dir ../../detector/data/kitti/training/log/
cd ../../detector/data/kitti/training/
mkdir score_hma
mv log/image_2 score_hma/
mv log/image_3 score_hma/
rm -r log
cd $CURPATH