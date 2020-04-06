#!/bin/bash
#$ -t 1-2
#$ -l m_mem_free=5G
#$ -N p_discrim
#$ -j y
#$ -o logs/assess/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/assess/p_discrim.py --name listings
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/assess/p_discrim.py --name threads
fi