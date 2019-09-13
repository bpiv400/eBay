#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=10G
#$ -t 1-784
#$ -N byr_round
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model byr --outcome round --id "$SGE_TASK_ID"