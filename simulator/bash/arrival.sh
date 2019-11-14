#!/bin/bash
#$ -l m_mem_free=60G
#$ -t 1-1296
#$ -N arrival
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model arrival --id "$SGE_TASK_ID"