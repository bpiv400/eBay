#!/bin/bash
#$ -t 1-6
#$ -l m_mem_free=50G
#$ -N f_init_value
#$ -j y
#$ -o logs/processing/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/processing/f_discrim/init_value.py --part test_rl --role slr
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/processing/f_discrim/init_value.py --part train_rl --role slr
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/processing/f_discrim/init_value.py --part test --role slr
fi