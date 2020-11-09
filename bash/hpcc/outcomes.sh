#!/bin/bash
#$ -t 1-512
#$ -q short.q
#$ -l m_mem_free=2G
#$ -N values
#$ -j y
#$ -o logs/outcomes/

ulimit -n 4096
python repo/sim/sims_array.py --num "$SGE_TASK_ID" --part "$1"