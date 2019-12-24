#!/bin/bash
#$ -N con_byr
#$ -o logs/train/
#$ -j y

python repo/models/train.py --name con_byr