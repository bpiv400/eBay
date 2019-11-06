#!/bin/bash
#$ -t 1-7627
#$ -q short.q
#$ -l m_mem_free=8G
#$ -N feats_cat
#$ -j y
#$ -o logs/
python repo/processing/b_feats/category.py --num "$SGE_TASK_ID"
