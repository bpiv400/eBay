#!/bin/bash
#$ -N hist
#$ -o logs/train/
#$ -j y

python repo/models/train.py --name hist