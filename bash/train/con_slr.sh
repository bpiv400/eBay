#!/bin/bash
#$ -N con_slr
#$ -o logs/train/
#$ -j y

python repo/models/train.py --name con_slr