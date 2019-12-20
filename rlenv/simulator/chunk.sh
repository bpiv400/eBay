#!/bin/bash
#$ -q all.q
#$ -t 1-3
#$ -l m_mem_free=50G
#$ -N chunk
#$ -j y
#$ -o logs/env/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/rlenv/simulator/chunk.py --part train_models
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/rlenv/simulator/chunk.py --part train_rl
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/rlenv/simulator/chunk.py --part test
fi