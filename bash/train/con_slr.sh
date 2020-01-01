#!/bin/bash
#$ -N con_slr
#$ -o logs/train/
#$ -j y

python repo/model/train.py --name con_slr