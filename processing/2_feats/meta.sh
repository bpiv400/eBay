#!/bin/bash
#$ -t 1-35
#$ -q all.q
#$ -l m_mem_free=12G
#$ -N feats_meta
#$ -j y
#$ -o logs/

python repo/processing/2_feats/meta.py --num "$SGE_TASK_ID"
