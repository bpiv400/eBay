#!/bin/bash
#$ -t 1-185
#$ -q short.q
#$ -l m_mem_free=12G
#$ -N c_cat
#$ -j y
#$ -o logs/processing/

python repo/processing/c_feats/category.py --num "$SGE_TASK_ID"
