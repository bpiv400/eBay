#!/bin/bash
#$ -t 1-512
#$ -q short.q
#$ -l m_mem_free=2G
#$ -N agent
#$ -j y
#$ -o logs/sims/

ulimit -n 4096
if [ "$1" == "heuristic" ]; then
  python repo/agent/eval/sims.py --num "$SGE_TASK_ID" --heuristic
else
  python repo/agent/eval/sims.py --num "$SGE_TASK_ID"
fi