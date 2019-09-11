#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=50G
#$ -t 1-84
#$ -N arrival_hist
#$ -o logs/

python repo/simulator/train.py --model arrival --outcome hist --id "$SGE_TASK_ID"