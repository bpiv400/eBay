#!/bin/bash
#$ -t 1-2
#$ -l m_mem_free=50G
#$ -N e_discrim
#$ -j y
#$ -o logs/processing/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/processing/e_inputs/listings.py --part "$1"
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/processing/e_inputs/threads.py --part "$1"
fi