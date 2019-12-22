#!/bin/bash
#$ -t 1-1000
#$ -q short.q
#$ -l m_mem_free=4G
#$ -N train_models
#$ -j y
#$ -o logs/discrim/

python repo/rlenv/simulator/generate.py --num "$SGE_TASK_ID" --part train_models
