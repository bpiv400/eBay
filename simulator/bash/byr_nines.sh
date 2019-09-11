#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=50G
#$ -t 1-2352
#$ -N byr_nines
#$ -o logs/

python repo/simulator/train.py --model byr --outcome nines --id "$SGE_TASK_ID"