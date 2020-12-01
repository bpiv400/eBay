#!/bin/bash
#$ -l m_mem_free=25G
#$ -N agent
#$ -j y
#$ -o logs/collate/

ulimit -n 4096
if [ "$2" == "heuristic" ]; then
  python repo/agent/eval/collate.py --delta "$1" --heuristic
else
  python repo/agent/eval/collate.py --delta "$1"
fi