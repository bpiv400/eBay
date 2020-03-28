#!/bin/bash
#$ -t 1-3
#$ -l m_mem_free=50G
#$ -N f_discrim
#$ -j y
#$ -o logs/processing/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/processing/f_discrim/listings.py --part "$1"
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/processing/f_discrim/threads.py --part "$1" --tf
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/processing/f_discrim/threads.py --part "$1"
fi