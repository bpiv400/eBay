#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=40G
#$ -t 1-196
#$ -N arrival_sec
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model arrival --outcome sec --id "$SGE_TASK_ID"