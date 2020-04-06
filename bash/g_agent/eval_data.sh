#!/bin/bash
#$ -t 1-1000
#$ -q short.q
#$ -l m_mem_free=4G
#$ -N discrim
#$ -j y
#$ -o logs/discrim/

python repo/agent/a_inputs/eval_chunks.py --num "$SGE_TASK_ID"