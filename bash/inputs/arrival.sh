#!/bin/bash
#$ -t 1-4
#$ -l m_mem_free=50G
#$ -N arrival
#$ -j y
#$ -o logs/inputs/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/inputs/first_arrival.py --part "$1"
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/inputs/next_arrival.py --part "$1"
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/inputs/hist.py --part "$1"
elif [ "$SGE_TASK_ID" == 4 ]
then
	python repo/inputs/arrival.py --part "$1"
fi