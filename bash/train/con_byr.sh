#!/bin/bash
#$ -N con_byr
#$ -o logs/train/
#$ -j y

python repo/train/train_model.py --name con_byr