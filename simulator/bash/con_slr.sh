#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=125G
#$ -t 1-28
#$ -N con_slr
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model con_slr --id "$SGE_TASK_ID"