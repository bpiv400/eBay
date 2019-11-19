#!/bin/bash
#$ -t 1-144
#$ -N con_byr
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model con_byr --id "$SGE_TASK_ID"