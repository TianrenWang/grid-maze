#!/bin/bash

python train.py "$1" "$2" --selfLocalize --numLearn 3000
python train.py "$1" "$2" --grid
