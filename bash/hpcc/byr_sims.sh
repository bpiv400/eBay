#!/bin/bash
#$ -t 1-512
#$ -q short.q
#$ -l m_mem_free=2G
#$ -N agent
#$ -j y
#$ -o logs/sims/

ulimit -n 4096
if [ "$2" == "heuristic" ]; then
  python repo/agent/eval/sims.py --byr --delta "$1" --num "$SGE_TASK_ID" --heuristic
elif [ "$2" != "" ]; then
  if [ "$3" == "heuristic" ]; then
    python repo/agent/eval/sims.py --byr --delta "$1" --turn_cost "$2" --num "$SGE_TASK_ID" --heuristic
  else
    python repo/agent/eval/sims.py --byr --delta "$1" --turn_cost "$2" --num "$SGE_TASK_ID"
  fi
  python repo/agent/eval/sims.py --byr --delta "$1" --num "$SGE_TASK_ID"
fi