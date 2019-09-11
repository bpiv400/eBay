#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=50G
#$ -t 1-588
#$ -N arrival_sec
#$ -o logs/

python repo/simulator/train.py --model arrival --outcome sec --id "$SGE_TASK_ID"