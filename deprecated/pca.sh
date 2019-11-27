#!/bin/bash
#$ -q all.q
#$ -t 1-3
#$ -l m_mem_free=180G
#$ -N frames_pca
#$ -j y
#$ -o logs/

python repo/processing/d_frames/pca.py --num "$SGE_TASK_ID"