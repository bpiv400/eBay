#!/bin/bash
#$ -t 1-3
#$ -l m_mem_free=25G
#$ -N f_no_arrival
#$ -j y
#$ -o logs/processing/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/processing/f_discrim/listings.py --part test_rl
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/processing/f_discrim/threads.py --part train_rl
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/processing/f_discrim/threads.py --part test
fi