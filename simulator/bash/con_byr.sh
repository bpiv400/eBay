#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=16G
#$ -t 1-28
#$ -N con_byr
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model con_byr --id "$SGE_TASK_ID"