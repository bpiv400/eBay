#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=40G
#$ -t 1-5488
#$ -N byr_con
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model byr --outcome con --id "$SGE_TASK_ID"