#!/bin/bash
#$ -l m_mem_free=25G
#$ -N agent
#$ -j y
#$ -o logs/collate/

ulimit -n 4096
if [ "$1" == "heuristic" ]; then
  python repo/agent/eval/collate.py --heuristic
else
  python repo/agent/eval/collate.py
fi