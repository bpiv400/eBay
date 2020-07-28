#!/bin/bash
#$ -t 1-3
#$ -l m_mem_free=50G
#$ -N discrim
#$ -j y
#$ -o logs/inputs/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/inputs/discrim.py --name listings --part "$1"
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/inputs/discrim.py --name threads --part "$1"
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/inputs/discrim.py --name threads_no_tf --part "$1"
fi