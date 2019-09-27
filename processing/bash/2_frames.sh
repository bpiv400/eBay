#!/bin/bash
#$ -t 1-256
#$ -q all.q
#$ -l m_mem_free=10G
#$ -N frames
#$ -j y
#$ -o logs/

python repo/processing/2_frames.py --num "$SGE_TASK_ID"
