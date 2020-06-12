#!/bin/bash
#$ -t 1-4
#$ -l m_mem_free=75G
#$ -N global
#$ -j y
#$ -o logs/processing/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/processing/a_global/clean.py
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/processing/a_global/partition.py
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/processing/a_global/w2v.py
elif [ "$SGE_TASK_ID" == 4 ]
then
	python repo/processing/a_global/w2v.py --slr
fi