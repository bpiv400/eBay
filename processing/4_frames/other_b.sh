#!/bin/bash
#$ -q all.q
#$ -t 1-3
#$ -l m_mem_free=150G
#$ -N other_b
#$ -j y
#$ -o logs/

python repo/processing/4_frames/other_b.py --num "$SGE_TASK_ID"