#!/bin/bash
#$ -t 1-3
#$ -l m_mem_free=150G
#$ -N days
#$ -j y
#$ -o logs/

python repo/processing/5_inputs/days.py --num "$SGE_TASK_ID"
