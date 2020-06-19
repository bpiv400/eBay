#!/bin/bash
#$ -t 1-256
#$ -q short.q
#$ -l m_mem_free=4G
#$ -N outcomes
#$ -j y
#$ -o logs/sim/

python repo/sim/generate.py --name outcomes --num "$SGE_TASK_ID" --part "$1"
