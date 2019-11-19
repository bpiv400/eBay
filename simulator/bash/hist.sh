#!/bin/bash
#$ -t 1-48
#$ -N hist
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model hist --id "$SGE_TASK_ID"