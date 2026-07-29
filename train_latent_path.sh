#!/bin/bash

python train.py "$1" "$2" --pretrain --numLearn 2000
python train.py "$1" "$2" --pretrain --numLearn 1500 --entropy 0.01
