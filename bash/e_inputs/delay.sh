#!/bin/bash
#$ -t 2-7
#$ -l m_mem_free=10G
#$ -N e_delay
#$ -j y
#$ -o logs/processing/

if [ "$SGE_TASK_ID" == 2 ]
then
	python repo/processing/e_inputs/offer.py --part "$1" --outcome delay --turn 2
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/processing/e_inputs/offer.py --part "$1" --outcome delay --turn 3
elif [ "$SGE_TASK_ID" == 4 ]
then
	python repo/processing/e_inputs/offer.py --part "$1" --outcome delay --turn 4
elif [ "$SGE_TASK_ID" == 5 ]
then
	python repo/processing/e_inputs/offer.py --part "$1" --outcome delay --turn 5
elif [ "$SGE_TASK_ID" == 6 ]
then
	python repo/processing/e_inputs/offer.py --part "$1" --outcome delay --turn 6
elif [ "$SGE_TASK_ID" == 7 ]
then
	python repo/processing/e_inputs/offer.py --part "$1" --outcome delay --turn 7
fi