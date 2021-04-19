#!/bin/bash
#$ -t 1-512
#$ -q short.q
#$ -l m_mem_free=2G
#$ -N slr_sims
#$ -j y
#$ -o logs/sims/

ulimit -n 4096
if [ "$3" == "heuristic" ]; then
  python repo/agent/eval/sims.py --num "$SGE_TASK_ID" --delta "$1" --heuristic
else
  python repo/agent/eval/sims.py --num "$SGE_TASK_ID" --delta "$1"
fi