#!/bin/bash
#$ -t 1-1024
#$ -q short.q
#$ -l m_mem_free=2G
#$ -N sims
#$ -j y
#$ -o logs/sims/

ulimit -n 4096
if [ "$2" == "values" ]; then
  python repo/sim/sims.py --num "$SGE_TASK_ID" --part "$1" --values
else
  python repo/sim/sims.py --num "$SGE_TASK_ID" --part "$1"
fi