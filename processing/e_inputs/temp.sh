#!/bin/bash
#$ -t 1-5
#$ -l m_mem_free=20G
#$ -N hist
#$ -j y
#$ -o logs/

python repo/processing/e_inputs/temp.py --num "$SGE_TASK_ID"