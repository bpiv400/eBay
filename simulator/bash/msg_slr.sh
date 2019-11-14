#!/bin/bash
#$ -l m_mem_free=60G
#$ -t 1-1296
#$ -N msg_slr
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model msg_slr --id "$SGE_TASK_ID"