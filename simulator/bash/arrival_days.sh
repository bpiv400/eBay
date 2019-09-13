#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=10G
#$ -t 1-140
#$ -N arrival_days
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model arrival --outcome days --id "$SGE_TASK_ID"