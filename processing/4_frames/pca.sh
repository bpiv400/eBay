#!/bin/bash
#$ -q all.q
#$ -t 1-2
#$ -l m_mem_free=150G
#$ -N pca
#$ -j y
#$ -o logs/

python repo/processing/4_frames/pca.py --num "$SGE_TASK_ID"