#!/bin/bash
#$ -t 1-256
#$ -q short.q
#$ -l m_mem_free=10G
#$ -N feats
#$ -j y
#$ -o logs/

python repo/processing/3_feats.py --num "$SGE_TASK_ID"