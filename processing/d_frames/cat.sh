#!/bin/bash
#$ -q all.q
#$ -t 1-3
#$ -l m_mem_free=75G
#$ -N frames_cat
#$ -j y
#$ -o logs/

python repo/processing/d_frames/cat.py --num "$SGE_TASK_ID"