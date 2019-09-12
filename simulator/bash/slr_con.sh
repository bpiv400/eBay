#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=30G
#$ -t 1-5488
#$ -N slr_con
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model slr --outcome con --id "$SGE_TASK_ID"