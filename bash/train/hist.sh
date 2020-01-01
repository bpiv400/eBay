#!/bin/bash
#$ -N hist
#$ -o logs/train/
#$ -j y

python repo/model/train.py --name hist