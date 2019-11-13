#!/bin/bash
#$ -t 1-12
#$ -l m_mem_free=50G
#$ -N offer
#$ -j y
#$ -o logs/

python repo/processing/e_inputs/offer.py --num "$SGE_TASK_ID"
