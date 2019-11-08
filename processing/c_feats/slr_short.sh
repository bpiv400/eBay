#!/bin/bash
#$ -t 1-835
#$ -q short.q
#$ -l m_mem_free=3G
#$ -N feats_slr
#$ -j y
#$ -o logs/

python repo/processing/b_feats/slr.py --num "$SGE_TASK_ID"
