#!/bin/bash
#$ -t 1-6
#$ -l m_mem_free=125G
#$ -N e_discrim
#$ -j y
#$ -o logs/processing

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/processing/e_inputs/listings.py --part train_models
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/processing/e_inputs/threads.py --part train_models
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/processing/e_inputs/listings.py --part train_rl
elif [ "$SGE_TASK_ID" == 4 ]
then
	python repo/processing/e_inputs/threads.py --part train_rl
elif [ "$SGE_TASK_ID" == 5 ]
then
	python repo/processing/e_inputs/listings.py --part test
elif [ "$SGE_TASK_ID" == 6 ]
then
	python repo/processing/e_inputs/threads.py --part test