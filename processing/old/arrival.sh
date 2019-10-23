#!/bin/bash
#$ -t 1-6
#$ -l m_mem_free=50G
#$ -N arrival
#$ -j y
#$ -o logs/

python repo/processing/5_inputs/arrival.py --num "$SGE_TASK_ID"
