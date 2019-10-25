#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=16G
#$ -t 1-28
#$ -N arrival
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model arrival --id "$SGE_TASK_ID"