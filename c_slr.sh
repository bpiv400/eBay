#!/bin/bash
#$ -t 1-141
#$ -q short.q
#$ -l m_mem_free=8G
#$ -N c_slr
#$ -j y
#$ -o logs/processing/

python repo/processing/c_feats/category.py --num "$SGE_TASK_ID" --slr

