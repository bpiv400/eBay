#!/bin/bash
#$ -l m_mem_free=25G
#$ -N byr_collate_h
#$ -j y
#$ -o logs/collate/

ulimit -n 4096
if [ "$2" == "" ]; then
  python repo/agent/eval/collate.py --byr --index "$1" --heuristic
else
  python repo/agent/eval/collate.py --byr --index "$1" --part "$2" --heuristic
fi