#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=10G
#$ -t 1-140
#$ -N byr_delay
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model byr --outcome delay --id "$SGE_TASK_ID"