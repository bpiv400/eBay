#!/bin/bash
#$ -q all.q
#$ -t 1-2
#$ -l m_mem_free=100G
#$ -N pca
#$ -j y
#$ -o logs/

python repo/processing/4_pca/pca.py --num "$SGE_TASK_ID"