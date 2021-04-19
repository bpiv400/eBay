#!/bin/bash
#$ -l m_mem_free=25G
#$ -N slr_collate
#$ -j y
#$ -o logs/collate/

ulimit -n 4096
if [ "$3" == "heuristic" ]; then
  python repo/agent/eval/collate.py --delta "$1" --heuristic
else
  python repo/agent/eval/collate.py --delta "$1"
fi