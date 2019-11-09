#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=125G
#$ -t 1-28
#$ -N msg_byr
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model msg_byr --id "$SGE_TASK_ID"