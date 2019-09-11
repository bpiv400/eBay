#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=50G
#$ -t 1-2352
#$ -N slr_round
#$ -o logs/

python repo/simulator/train.py --model slr --outcome round --id "$SGE_TASK_ID"