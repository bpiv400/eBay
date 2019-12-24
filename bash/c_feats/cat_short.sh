#!/bin/bash
#$ -t 1-514
#$ -q short.q
#$ -l m_mem_free=12G
#$ -N feats_cat
#$ -j y
#$ -o logs/
python repo/processing/c_feats/category.py --num "$SGE_TASK_ID"
