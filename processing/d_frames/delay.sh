#!/bin/bash
#$ -q all.q
#$ -t 1-6
#$ -l m_mem_free=150G
#$ -N frames_delay
#$ -j y
#$ -o logs/

python repo/processing/d_frames/delay.py --num "$SGE_TASK_ID"