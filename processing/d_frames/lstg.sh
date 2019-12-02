#!/bin/bash
#$ -q all.q
#$ -t 1-3
#$ -l m_mem_free=75G
#$ -N frames_lstg
#$ -j y
#$ -o logs/

python repo/processing/d_frames/lstg.py --num "$SGE_TASK_ID"