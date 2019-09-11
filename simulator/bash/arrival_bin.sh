#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=50G
#$ -t 1-84
#$ -N arrival_bin
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model arrival --outcome bin --id "$SGE_TASK_ID"