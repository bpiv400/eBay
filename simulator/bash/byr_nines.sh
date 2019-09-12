#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=40G
#$ -t 1-784
#$ -N byr_nines
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model byr --outcome nines --id "$SGE_TASK_ID"