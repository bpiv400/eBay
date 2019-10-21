#!/bin/bash
#$ -t 1-3
#$ -l m_mem_free=50G
#$ -N concession
#$ -j y
#$ -o logs/

python repo/processing/e_inputs/concession.py --num "$SGE_TASK_ID"
