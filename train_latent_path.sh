#!/bin/bash

python train.py "$1" "$2" --pretrain --numLearn 1200
python train.py "$1" "$2" --learnManifold
