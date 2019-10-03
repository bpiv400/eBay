#!/bin/bash
#$ -q all.q
#$ -t 1-3
#$ -l m_mem_free=100G
#$ -N other
#$ -j y
#$ -o logs/

python repo/processing/4_frames/other.py --num "$SGE_TASK_ID"