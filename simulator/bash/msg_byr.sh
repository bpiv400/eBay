#!/bin/bash
#$ -t 1-48
#$ -N msg_byr
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model msg_byr --id "$SGE_TASK_ID"