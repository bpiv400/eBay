#!/bin/bash
#$ -t 1-6
#$ -l m_mem_free=50G
#$ -N e_msg
#$ -j y
#$ -o logs/processing/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/processing/e_inputs/offer.py --part "$1" --outcome msg --turn 1
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/processing/e_inputs/offer.py --part "$1" --outcome msg --turn 2
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/processing/e_inputs/offer.py --part "$1" --outcome msg --turn 3
elif [ "$SGE_TASK_ID" == 4 ]
then
	python repo/processing/e_inputs/offer.py --part "$1" --outcome msg --turn 4
elif [ "$SGE_TASK_ID" == 5 ]
then
	python repo/processing/e_inputs/offer.py --part "$1" --outcome msg --turn 5
elif [ "$SGE_TASK_ID" == 6 ]
then
	python repo/processing/e_inputs/offer.py --part "$1" --outcome msg --turn 6
fi