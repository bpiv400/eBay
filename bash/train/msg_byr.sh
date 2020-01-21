#!/bin/bash
#$ -N msg_byr
#$ -o logs/train/
#$ -j y

python repo/train/train_model.py --name msg_byr