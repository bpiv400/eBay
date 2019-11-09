#!/bin/bash
#$ -t 1-835
#$ -q short.q
#$ -l m_mem_free=10G
#$ -N feats_slr
#$ -j y
#$ -o logs/

python repo/processing/c_feats/category.py --num "$SGE_TASK_ID" --slr