#!/bin/bash
#$ -t 1-1024
#$ -q short.q
#$ -l m_mem_free=2G
#$ -N chunks
#$ -j y
#$ -o logs/chunks/

ulimit -n 4096
python repo/sim/chunks.py --num "$SGE_TASK_ID" --part "$1"