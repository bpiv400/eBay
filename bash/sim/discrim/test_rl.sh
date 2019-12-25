#!/bin/bash
#$ -t 1-1000
#$ -q short.q
#$ -l m_mem_free=4G
#$ -N test_rl
#$ -j y
#$ -o logs/discrim/

python repo/rlenv/simulator/generate.py --num "$SGE_TASK_ID" --part test_rl