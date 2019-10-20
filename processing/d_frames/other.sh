#!/bin/bash
#$ -q all.q
#$ -t 1-3
#$ -l m_mem_free=150G
#$ -N other
#$ -j y
#$ -o logs/

python repo/processing/d_frames/other.py --num "$SGE_TASK_ID"