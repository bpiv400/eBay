#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=50G
#$ -o logs/
#$ -t 1-588
#$ -N arrival_sec

python repo/simulator/train.py --model arrival --outcome sec --id "$SGE_TASK_ID"