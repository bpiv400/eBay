#!/bin/bash
#$ -t 1-3
#$ -l m_mem_free=50G
#$ -N e_arrival
#$ -j y
#$ -o logs/processing/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/processing/e_inputs/first_arrival.py --part "$1"
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/processing/e_inputs/next_arrival.py --part "$1"
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/processing/e_inputs/hist.py --part "$1"
fi