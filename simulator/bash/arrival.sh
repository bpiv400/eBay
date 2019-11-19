#!/bin/bash
#$ -t 1-48
#$ -N arrival
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model arrival --id "$SGE_TASK_ID"