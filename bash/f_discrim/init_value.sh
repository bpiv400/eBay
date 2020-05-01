#!/bin/bash
#$ -t 1-6
#$ -l m_mem_free=25G
#$ -N f_init_value
#$ -j y
#$ -o logs/processing/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/processing/f_discrim/init_value.py --part test_rl
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/processing/f_discrim/init_value.py --part train_rl
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/processing/f_discrim/init_value.py --part test
fi