#!/bin/bash
#$ -t 1-10000
#$ -q short.q
#$ -N train
#$ -j y
#$ -o logs/

python repo/simulator/train.py --model "$1" --outcome "$2" --id "$SGE_TASK_ID"
