#!/bin/bash
#$ -t 1-1024
#$ -q short.q
#$ -l m_mem_free=2G
#$ -N byr_sims_2
#$ -j y
#$ -o logs/sims/

ulimit -n 4096
if [ "$2" == "" ]; then
  python repo/agent/eval/sims.py --byr --delta "$1" --agent_thread 2 --num "$SGE_TASK_ID"
else
  python repo/agent/eval/sims.py --byr --delta "$1" --turn_cost "$2" --agent_thread 2 --num "$SGE_TASK_ID"
fi