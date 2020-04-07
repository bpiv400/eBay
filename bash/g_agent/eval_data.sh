#!/bin/bash
#$ -t 1-1000
#$ -q short.q
#$ -l m_mem_free=4G
#$ -N rl_eval_chunks
#$ -j y
#$ -o logs/

python repo/agent/a_inputs/eval_data.py --num "$SGE_TASK_ID"