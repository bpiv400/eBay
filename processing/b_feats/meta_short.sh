#!/bin/bash
#$ -t 1-245
#$ -q short.q
#$ -l m_mem_free=75G
#$ -N feats_meta
#$ -j y
#$ -o logs/
TASK_ID=$((SGE_TASK_ID - 1))
NUM=$((TASK_ID / 7 + 1))
FEAT=$((TASK_ID % 7 + 1))
python repo/processing/b_feats/meta.py --num "$NUM" --feat "$FEAT"
