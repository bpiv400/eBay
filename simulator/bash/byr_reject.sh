#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=50G
#$ -t 1-2352
#$ -N byr_reject
#$ -o logs/

python repo/simulator/train.py --model byr --outcome reject --id "$SGE_TASK_ID"