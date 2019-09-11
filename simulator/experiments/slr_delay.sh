#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=50G
#$ -o logs/
#$ -t 1-2352
#$ -N slr_delay

python repo/simulator/train.py --model slr --outcome delay --id "$SGE_TASK_ID"