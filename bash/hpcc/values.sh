#!/bin/bash
#$ -t 1-4096
#$ -q short.q
#$ -l m_mem_free=2G
#$ -N values
#$ -j y
#$ -o logs/values/

python repo/sim/values_server.py --num "$SGE_TASK_ID" --part "$1"