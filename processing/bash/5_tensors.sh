#!/bin/bash
#$ -t 1-19
#$ -l m_mem_free=50G
#$ -N tensors
#$ -j y
#$ -o logs/

python repo/processing/5_tensors.py --id "$SGE_TASK_ID"
