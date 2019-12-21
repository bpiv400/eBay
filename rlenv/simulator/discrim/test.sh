#!/bin/bash
#$ -t 1-100000
#$ -q short.q
#$ -l m_mem_free=4G
#$ -N test
#$ -j y
#$ -o logs/discrim/

python repo/rlenv/simulator/generate.py --num "$SGE_TASK_ID" --part test
