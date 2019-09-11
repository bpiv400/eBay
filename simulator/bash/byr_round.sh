#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=50G
#$ -t 1-2352
#$ -N byr_round
#$ -o logs/

python repo/simulator/train.py --model byr --outcome round --id "$SGE_TASK_ID"