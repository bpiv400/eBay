#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=250G
#$ -N frames_lstg
#$ -j y
#$ -o logs/

python repo/processing/d_frames/lstg.py