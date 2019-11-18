#!/bin/bash
#$ -pe openmp 2
#$ -t 1-144
#$ -N hist
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model hist --id "$SGE_TASK_ID"