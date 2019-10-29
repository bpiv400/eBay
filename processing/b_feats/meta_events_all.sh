#!/bin/bash
#$ -t 1-35
#$ -q all.q
#$ -l m_mem_free=32G
#$ -N feats_meta
#$ -j y
#$ -o logs/
python repo/processing/b_feats/meta_events.py --num "$SGE_TASK_ID"
