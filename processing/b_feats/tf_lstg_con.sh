#!/bin/bash
#$ -t 1-835
#$ -q short.q
#$ -l m_mem_free=10G
#$ -N tf_lstg
#$ -j y
#$ -o logs/

python repo/processing/b_feats/tf_lstg.py --num "$SGE_TASK_ID"
