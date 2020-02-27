#!/bin/bash
#$ -q all.q
#$ -t 1-3
#$ -l m_mem_free=50G
#$ -N chunk_env
#$ -j y
#$ -o logs/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/agent/a_inputs/seller_inputs.py --part train_rl
elif [ "$SGE_TASK_ID" == 2 ]
thend
	python repo/agent/a_inputs/seller_inputs.py --part test_rl
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/agent/a_inputs/seller_inputs.py --part train_models
fi