#!/bin/bash
#$ -N con_slr
#$ -o logs/train/
#$ -j y

python repo/train/train_model.py --name con_slr