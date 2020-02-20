#!/bin/bash
#$ -t 1-11
#$ -l m_mem_free=20G
#$ -N e_train_rl
#$ -j y
#$ -o logs/processing/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/processing/e_inputs/first_arrival.py --part train_rl
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/processing/e_inputs/next_arrival.py --part train_rl
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/processing/e_inputs/hist.py --part train_rl
elif [ "$SGE_TASK_ID" == 4 ]
then
	python repo/processing/e_inputs/first_offer.py --part train_rl --outcome con
elif [ "$SGE_TASK_ID" == 5 ]
then
	python repo/processing/e_inputs/first_offer.py --part train_rl --outcome msg
elif [ "$SGE_TASK_ID" == 6 ]
then
	python repo/processing/e_inputs/next_offer.py --part train_rl --outcome delay --role byr
elif [ "$SGE_TASK_ID" == 7 ]
then
	python repo/processing/e_inputs/next_offer.py --part train_rl --outcome delay --role slr
elif [ "$SGE_TASK_ID" == 8 ]
then
	python repo/processing/e_inputs/next_offer.py --part train_rl --outcome con --role byr
elif [ "$SGE_TASK_ID" == 9 ]
then
	python repo/processing/e_inputs/next_offer.py --part train_rl --outcome con --role slr
elif [ "$SGE_TASK_ID" == 10 ]
then
	python repo/processing/e_inputs/next_offer.py --part train_rl --outcome msg --role byr
elif [ "$SGE_TASK_ID" == 11 ]
then
	python repo/processing/e_inputs/next_offer.py --part train_rl --outcome msg --role slr
fi