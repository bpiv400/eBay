#!/bin/bash
#$ -t 1-835
#$ -q all.q
#$ -l m_mem_free=10G
#$ -N feats_slr
#$ -j y
#$ -o logs/

python repo/processing/b_feats/slr.py --num "$SGE_TASK_ID"
