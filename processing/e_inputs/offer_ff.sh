#!/bin/bash
#$ -t 1-12
#$ -l m_mem_free=50G
#$ -N offer_ff
#$ -j y
#$ -o logs/

python repo/processing/e_inputs/offer_ff.py --num "$SGE_TASK_ID"
