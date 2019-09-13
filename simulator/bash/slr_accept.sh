#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=10G
#$ -t 1-784
#$ -N slr_accept
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model slr --outcome accept --id "$SGE_TASK_ID"