#!/bin/bash
#$ -t 1-512
#$ -q short.q
#$ -l m_mem_free=8G
#$ -N env_sim_discrim
#$ -j y
#$ -o logs/

python repo/rlenv/simulator/generate.py --num "$SGE_TASK_ID" --part "$1"
