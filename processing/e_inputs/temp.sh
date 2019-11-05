#!/bin/bash
#$ -t 1-18
#$ -l m_mem_free=75G
#$ -N temp
#$ -j y
#$ -o logs/

python repo/processing/e_inputs/temp.py --num "$SGE_TASK_ID"
