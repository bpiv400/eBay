#!/bin/bash
#$ -t 1-8
#$ -l m_mem_free=100G
#$ -N e_train_models
#$ -j y
#$ -o logs/processing/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/processing/e_inputs/arrival.py --part train_models
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/processing/e_inputs/hist.py --part train_models
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/processing/e_inputs/delay.py --part train_models --role byr
elif [ "$SGE_TASK_ID" == 4 ]
then
	python repo/processing/e_inputs/delay.py --part train_models --role slr
elif [ "$SGE_TASK_ID" == 5 ]
then
	python repo/processing/e_inputs/offer.py --part train_models --outcome con --role byr
elif [ "$SGE_TASK_ID" == 6 ]
then
	python repo/processing/e_inputs/offer.py --part train_models --outcome con --role slr
elif [ "$SGE_TASK_ID" == 7 ]
then
	python repo/processing/e_inputs/offer.py --part train_models --outcome msg --role byr
elif [ "$SGE_TASK_ID" == 8 ]
then
	python repo/processing/e_inputs/offer.py --part train_models --outcome msg --role slr
fi