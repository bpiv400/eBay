#!/bin/bash
#$ -t 1-6
#$ -l m_mem_free=75G
#$ -N delay
#$ -j y
#$ -o logs/

python repo/processing/e_inputs/delay.py --num "$SGE_TASK_ID"
