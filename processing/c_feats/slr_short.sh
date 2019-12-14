#!/bin/bash
#$ -t 1-512
#$ -q short.q
#$ -l m_mem_free=8G
#$ -N reward_test
#$ -j y
#$ -o logs/

python repo/processing/c_feats/category.py --num "$SGE_TASK_ID" --slr

