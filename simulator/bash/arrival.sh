#!/bin/bash
#$ -pe openmp 2
#$ -l m_mem_free=30G
#$ -t 1-144
#$ -N arrival
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model arrival --id "$SGE_TASK_ID"