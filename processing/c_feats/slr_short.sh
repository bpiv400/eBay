#!/bin/bash
#$ -t 1-300
#$ -q short.q
#$ -l m_mem_free=8G
#$ -N feats_slr
#$ -j y
#$ -o logs/

python repo/processing/c_feats/category.py --num "$SGE_TASK_ID" --slr

