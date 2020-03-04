#!/bin/bash
#$ -t 1-2
#$ -l m_mem_free=50G
#$ -N e_init
#$ -j y
#$ -o logs/processing/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/processing/e_inputs/initialize.py --part "$1" --role slr
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/processing/e_inputs/initialize.py --part "$1" --role byr
fi