#!/bin/bash
#$ -l m_mem_free=25G
#$ -N agent
#$ -j y
#$ -o logs/collate/

ulimit -n 4096
if [ "$3" == "heuristic" ]; then
  python repo/agent/eval/collate.py --byr --delta "$1" --turn_cost "$2" --heuristic
else
  python repo/agent/eval/collate.py --byr --delta "$1" --turn_cost "$2"
fi