#!/bin/bash
#$ -t 1-512
#$ -q short.q
#$ -l m_mem_free=2G
#$ -N agent
#$ -j y
#$ -o logs/sims/

ulimit -n 4096
python repo/agent/eval/sims.py --num "$SGE_TASK_ID" --delta "$1" "$2"