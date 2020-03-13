#!/bin/bash
#$ -N no_dropout
#$ -o logs/train/
#$ -j y

python repo/train/train_model.py --name "$1"