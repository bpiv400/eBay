#!/bin/bash
#$ -l m_mem_free=25G
#$ -N byr_collate_2
#$ -j y
#$ -o logs/collate/

ulimit -n 4096
if [ "$2" == "" ]; then
  python repo/agent/eval/collate.py --byr --delta "$1" --agent_thread 2 --num "$SGE_TASK_ID"
else
  python repo/agent/eval/collate.py --byr --delta "$1" --turn_cost "$2" --agent_thread 2 --num "$SGE_TASK_ID"
fi