#!/bin/bash
#$ -t 1-3
#$ -l m_mem_free=5G
#$ -N assess
#$ -j y
#$ -o logs/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/assess/training_curves.py
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/assess/distributions.py
elif [ "$SGE_TASK_ID" == 3 ]
then
  python repo/assess/roc.py
fi