#!/bin/bash
#$ -q all.q
#$ -t 1-3
#$ -l m_mem_free=25G
#$ -N frames_thread
#$ -j y
#$ -o logs/

python repo/processing/d_frames/thread.py --num "$SGE_TASK_ID"