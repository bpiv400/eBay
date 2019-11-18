#!/bin/bash
#$ -pe openmp 2
#$ -t 1-144
#$ -N arrival
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model arrival --id "$SGE_TASK_ID"