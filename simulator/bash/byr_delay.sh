#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=30G
#$ -t 1-784
#$ -N byr_delay
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model byr --outcome delay --id "$SGE_TASK_ID"