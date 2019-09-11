#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=50G
#$ -o logs/
#$ -t 1-84
#$ -N arrival_bin

python repo/simulator/train.py --model arrival --outcome bin --id "$SGE_TASK_ID"