#!/bin/bash
#$ -t 1-100000
#$ -q short.q
#$ -l m_mem_free=4G
#$ -N discrim
#$ -j y
#$ -o logs/env/

python repo/rlenv/simulator/generate.py --num "$SGE_TASK_ID" --part "$1"
