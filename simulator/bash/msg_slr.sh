#!/bin/bash
#$ -t 1-48
#$ -N msg_slr
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model msg_slr --id "$SGE_TASK_ID"