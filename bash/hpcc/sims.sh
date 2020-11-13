#!/bin/bash
#$ -t 1-512
#$ -q short.q
#$ -l m_mem_free=2G
#$ -N sims
#$ -j y
#$ -o logs/sims/

ulimit -n 4096
if [ "$1" == "values" ]; then
  python repo/sim/sims.py --num "$SGE_TASK_ID" --values
else
  python repo/sim/sims.py --num "$SGE_TASK_ID"
fi