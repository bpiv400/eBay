#!/bin/bash
#$ -t 1-9
#$ -l m_mem_free=120G
#$ -N e_train_models
#$ -j y
#$ -o logs/processing/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/processing/e_inputs/first_arrival.py --part train_models
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/processing/e_inputs/interarrival.py --part train_models
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/processing/e_inputs/hist.py --part train_models
elif [ "$SGE_TASK_ID" == 4 ]
then
	python repo/processing/e_inputs/offer.py --part train_models --outcome delay --role byr
elif [ "$SGE_TASK_ID" == 5 ]
then
	python repo/processing/e_inputs/offer.py --part train_models --outcome delay --role slr
elif [ "$SGE_TASK_ID" == 6 ]
then
	python repo/processing/e_inputs/offer.py --part train_models --outcome con --role byr
elif [ "$SGE_TASK_ID" == 7 ]
then
	python repo/processing/e_inputs/offer.py --part train_models --outcome con --role slr
elif [ "$SGE_TASK_ID" == 8 ]
then
	python repo/processing/e_inputs/offer.py --part train_models --outcome msg --role byr
elif [ "$SGE_TASK_ID" == 9 ]
then
	python repo/processing/e_inputs/offer.py --part train_models --outcome msg --role slr
fi