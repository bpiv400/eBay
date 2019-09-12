#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=40G
#$ -t 1-784
#$ -N byr_reject
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model byr --outcome reject --id "$SGE_TASK_ID"