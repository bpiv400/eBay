#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=50G
#$ -t 1-84
#$ -N arrival_days
#$ -o logs/

python repo/simulator/train.py --model arrival --outcome days --id "$SGE_TASK_ID"