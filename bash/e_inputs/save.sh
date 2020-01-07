#!/bin/bash
#$ -t 1-9
#$ -l m_mem_free=80G
#$ -N e_save
#$ -j y
#$ -o logs/processing

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/processing/e_inputs/save.py --part train_models --name arrival
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/processing/e_inputs/save.py --part train_rl --name arrival
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/processing/e_inputs/save.py --part test_rl --name arrival
elif [ "$SGE_TASK_ID" == 4 ]
then
	python repo/processing/e_inputs/save.py --part train_models --name delay_byr
elif [ "$SGE_TASK_ID" == 5 ]
then
	python repo/processing/e_inputs/save.py --part train_rl --name delay_byr
elif [ "$SGE_TASK_ID" == 6 ]
then
	python repo/processing/e_inputs/save.py --part test_rl --name delay_byr
elif [ "$SGE_TASK_ID" == 7 ]
then
	python repo/processing/e_inputs/save.py --part train_models --name delay_slr
elif [ "$SGE_TASK_ID" == 8 ]
then
	python repo/processing/e_inputs/save.py --part train_rl --name delay_slr
elif [ "$SGE_TASK_ID" == 9 ]
then
	python repo/processing/e_inputs/save.py --part test_rl --name delay_slr
fi