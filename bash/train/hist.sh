#!/bin/bash
#$ -N hist
#$ -o logs/train/
#$ -j y

python repo/train/train_model.py --name hist