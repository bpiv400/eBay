#!/bin/bash
#$ -t 1-256
#$ -q short.q
#$ -l m_mem_free=10G
#$ -N frames
#$ -j y
#$ -o logs/

python repo/processing/2a_frames.py --num "$SGE_TASK_ID"