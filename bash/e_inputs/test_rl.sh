#!/bin/bash
#$ -t 1-10
#$ -l m_mem_free=6G
#$ -N e_test_rl
#$ -j y
#$ -o logs/processing/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/processing/e_inputs/arrival.py --part test_rl
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/processing/e_inputs/hist.py --part test_rl
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/processing/e_inputs/offer.py --part test_rl --outcome delay --role byr
elif [ "$SGE_TASK_ID" == 4 ]
then
	python repo/processing/e_inputs/offer.py --part test_rl --outcome delay --role slr
elif [ "$SGE_TASK_ID" == 5 ]
then
	python repo/processing/e_inputs/offer.py --part test_rl --outcome con --role byr
elif [ "$SGE_TASK_ID" == 6 ]
then
	python repo/processing/e_inputs/offer.py --part test_rl --outcome con --role slr
elif [ "$SGE_TASK_ID" == 7 ]
then
	python repo/processing/e_inputs/offer.py --part test_rl --outcome msg --role byr
elif [ "$SGE_TASK_ID" == 8 ]
then
	python repo/processing/e_inputs/offer.py --part test_rl --outcome msg --role slr
elif [ "$SGE_TASK_ID" == 9 ]
then
	python repo/processing/e_inputs/listings.py --part test_rl
elif [ "$SGE_TASK_ID" == 10 ]
then
	python repo/processing/e_inputs/threads.py --part test_rl
fi