#!/bin/bash
#$ -q all.q
#$ -t 1-3
#$ -l m_mem_free=50G
#$ -N frames_lstg
#$ -j y
#$ -o logs/

python repo/processing/d_frames/lstg.py