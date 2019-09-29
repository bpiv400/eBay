#!/bin/bash
#$ -t 1-3
#$ -q all.q
#$ -l m_mem_free=150G
#$ -N pca
#$ -j y
#$ -o logs/

python repo/processing/3_recombine/pca.py --num "$SGE_TASK_ID"