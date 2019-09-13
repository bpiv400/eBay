#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=10G
#$ -t 1-28
#$ -N arrival_loc
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model arrival --outcome loc --id "$SGE_TASK_ID"