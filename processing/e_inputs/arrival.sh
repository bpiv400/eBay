#!/bin/bash
#$ -t 1-12
#$ -l m_mem_free=50G
#$ -N arrival
#$ -j y
#$ -o logs/

python repo/processing/e_inputs/arrival.py --num "$SGE_TASK_ID"
