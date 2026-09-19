#!/bin/bash

python train.py --expName "$1" --pretrain --numLearn 200
cp -r checkpoints/"$1" checkpoints/"$1"-pretrained
python train.py --expName "$1" --learnManifold
