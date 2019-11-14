#!/bin/bash
#$ -l m_mem_free=60G
#$ -t 1-1296
#$ -N hist
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model hist --id "$SGE_TASK_ID"