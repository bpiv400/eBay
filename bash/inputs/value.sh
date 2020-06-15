#!/bin/bash
#$ -t 1-3
#$ -l m_mem_free=50G
#$ -N value
#$ -j y
#$ -o logs/inputs/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/inputs/value_slr.py --part "$1"
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/inputs/value_slr.py --part "$1" --delay
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/inputs/value_byr.py --part "$1"
fi