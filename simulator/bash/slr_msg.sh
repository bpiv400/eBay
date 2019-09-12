#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=30G
#$ -t 1-784
#$ -N slr_msg
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model slr --outcome msg --id "$SGE_TASK_ID"