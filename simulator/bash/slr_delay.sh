#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=50G
#$ -t 1-2352
#$ -N slr_delay
#$ -o logs/

python repo/simulator/train.py --model slr --outcome delay --id "$SGE_TASK_ID"