#!/bin/bash
#$ -t 1-1024
#$ -q short.q
#$ -l m_mem_free=2G
#$ -N slr_sims_h
#$ -j y
#$ -o logs/sims/

ulimit -n 4096
python repo/agent/eval/sims.py --num "$SGE_TASK_ID" --delta "$1" --heuristic