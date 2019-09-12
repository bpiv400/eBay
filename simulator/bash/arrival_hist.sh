#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=40G
#$ -t 1-28
#$ -N arrival_hist
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model arrival --outcome hist --id "$SGE_TASK_ID"