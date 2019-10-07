#!/bin/bash
#$ -t 1-36
#$ -l m_mem_free=50G
#$ -N role
#$ -j y
#$ -o logs/

python repo/processing/5_inputs/role.py --num "$SGE_TASK_ID"
