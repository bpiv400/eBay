#!/bin/bash
#$ -q all.q
#$ -t 1-3
#$ -l m_mem_free=150G
#$ -N frames_offer
#$ -j y
#$ -o logs/

python repo/processing/d_frames/offer.py --num "$SGE_TASK_ID"