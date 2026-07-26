#!/bin/bash

python train.py "$1" "$2" --selfLocalize
python train.py "$1" "$2" --grid --visionPolicy --numLearn 2000
python train.py "$1" "$2" --grid --visionPolicy --numLearn 1000 --entropy 0.01
python train.py "$1" "$2" --grid --numLearn 500
python train.py "$1" "$2" --grid --numLearn 500 --entropy 0.01
