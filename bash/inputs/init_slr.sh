#!/bin/bash
#$ -t 1-2
#$ -l m_mem_free=50G
#$ -N init_slr
#$ -j y
#$ -o logs/inputs/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/inputs/init_policy.py --part "$1" --role slr
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/inputs/init_value.py --part "$1" --role slr
fi