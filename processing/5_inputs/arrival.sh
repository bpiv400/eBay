#!/bin/bash
#$ -t 1-12
#$ -l m_mem_free=150G
#$ -N arrival
#$ -j y
#$ -o logs/

python repo/processing/5_inputs/arrival.py --num "$SGE_TASK_ID"
