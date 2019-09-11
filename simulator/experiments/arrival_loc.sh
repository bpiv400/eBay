#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=50G
#$ -o logs/
#$ -t 1-84
#$ -N arrival_loc

python repo/simulator/train.py --model arrival --outcome loc --id "$SGE_TASK_ID"