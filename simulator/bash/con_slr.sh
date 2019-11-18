#!/bin/bash
#$ -pe openmp 2
#$ -l m_mem_free=30G
#$ -t 1-144
#$ -N con_slr
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model con_slr --id "$SGE_TASK_ID"