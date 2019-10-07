#!/bin/bash
#$ -t 1-6
#$ -l m_mem_free=150G
#$ -N delay
#$ -j y
#$ -o logs/

python repo/processing/5_inputs/delay.py --num "$SGE_TASK_ID"
