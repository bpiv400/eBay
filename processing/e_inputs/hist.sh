#!/bin/bash
#$ -t 1-3
#$ -l m_mem_free=100G
#$ -N hist
#$ -j y
#$ -o logs/

python repo/processing/e_inputs/hist.py --num "$SGE_TASK_ID"
