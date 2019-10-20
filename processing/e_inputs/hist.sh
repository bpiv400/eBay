#!/bin/bash
#$ -t 1-6
#$ -l m_mem_free=50G
#$ -N hist
#$ -j y
#$ -o logs/

python repo/processing/e_inputs/hist.py --num "$SGE_TASK_ID"
