#!/bin/bash
#$ -t 1-2
#$ -q all.q
#$ -l m_mem_free=75G
#$ -N w2v
#$ -j y
#$ -o logs/

python repo/processing/a_global/w2v.py --num "$SGE_TASK_ID"
