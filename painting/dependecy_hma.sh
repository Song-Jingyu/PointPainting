#!/bin/bash

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

git clone https://github.com/NVIDIA/semantic-segmentation
mv semantic-segmentation PointPainting/segmentation/hma/semantic-segmentation
cd PointPainting/segmentation/hma/semantic-segmentation
cp ../config.py ./
cp ../val.py ./
mkdir -p assets/seg_weights

pip install gdown tabulate runx coolname pyyaml
gdown --id 1lse0Mqf7ny5qqV99nGQ3ccXTKJ6kNGoH
mv cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth ./assets/seg_weights
gdown --id 1DTY2qu8N3vD-Yw7OAEEQORqzpotxSMZk 
mv hrnetv2_w48_imagenet_pretrained.pth ./assets/seg_weights

python val.py --eval_folder ../../training/ --result_dir ../../training/log/
cd ../../training/
mkdir score_hma
mv log/image_2 score_hma/
mv log/image_3 score_hma/
rm -r log
cd ..