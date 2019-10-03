#!/bin/bash
#$ -t 1-57
#$ -l m_mem_free=50G
#$ -N inputs
#$ -j y
#$ -o logs/

python repo/processing/5_inputs.py --id "$SGE_TASK_ID"
