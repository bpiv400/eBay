#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=100G
#$ -t 1-28
#$ -N hist
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model hist --id "$SGE_TASK_ID"