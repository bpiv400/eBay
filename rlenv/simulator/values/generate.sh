#!/bin/bash
#$ -t 1-10000
#$ -q short.q
#$ -l m_mem_free=8G
#$ -N env_sim_vals
#$ -j y
#$ -o logs/

python repo/rlenv/simulator/generate.py --num "$SGE_TASK_ID" --part "$1" --values
