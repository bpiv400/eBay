#!/bin/bash
#$ -t 1-20000
#$ -q short.q
#$ -l m_mem_free=75G
#$ -N feats_meta
#$ -j y
#$ -o logs/
python repo/processing/b_feats/category.py --num "$NUM" --feat "$SGE_TASK_ID"
