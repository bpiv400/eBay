#!/bin/bash
#$ -t 1-245
#$ -q all.q
#$ -l m_mem_free=75G
#$ -N feats_meta
#$ -j y
#$ -o logs/
NUM=$((SGE_TASK_ID / 7))
FEAT=$((7 - SGE_TASK_ID % 7))
python repo/processing/b_feats/meta.py --num "$NUM" --feat "$FEAT"
