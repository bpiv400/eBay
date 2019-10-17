#!/bin/bash
#$ -t 1-6
#$ -l m_mem_free=50G
#$ -N concession
#$ -j y
#$ -o logs/

python repo/processing/5_inputs/concession.py --num "$SGE_TASK_ID"
