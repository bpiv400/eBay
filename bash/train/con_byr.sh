#!/bin/bash
#$ -N con_byr
#$ -o logs/train/
#$ -j y

python repo/model/train.py --name con_byr