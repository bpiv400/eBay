#!/bin/bash
#$ -t 1-5845
#$ -q short.q
#$ -l m_mem_free=3G
#$ -N feats_slr
#$ -j y
#$ -o logs/
TASK_ID=$((SGE_TASK_ID - 1))
NUM=$((TASK_ID / 7 + 1))
FEAT=$((TASK_ID % 7 + 1))
python repo/processing/b_feats/slr.py --num "$NUM" --feat "$FEAT"
