#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=50G
#$ -t 1-84
#$ -N arrival_loc
#$ -o logs/

python repo/simulator/train.py --model arrival --outcome loc --id "$SGE_TASK_ID"