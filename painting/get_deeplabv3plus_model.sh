#!/bin/bash

CURPATH=$(cd "$(dirname "$0")"; pwd)
cd $CURPATH

wget https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth

cd mmseg
mkdir checkpoints
mv ../deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth ./checkpoints
cd ..
