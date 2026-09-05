#!/bin/bash

python train.py "$1" "$2" --pretrain --numLearn 600 --maxSteps 58
cp -r checkpoints/"$2" checkpoints/jepa-pretrained
python train.py "$1" "$2" --learnManifold --maxSteps 58
