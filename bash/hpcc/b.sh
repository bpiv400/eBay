#!/bin/bash
#$ -t 1-4
#$ -l m_mem_free=75G
#$ -N feats
#$ -j y
#$ -o logs/processing/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/processing/b_feats/category.py --part "$1" --name slr
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/processing/b_feats/category.py --part "$1" --name meta
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/processing/b_feats/category.py --part "$1" --name leaf
elif [ "$SGE_TASK_ID" == 4 ]
then
	python repo/processing/b_feats/tf.py --part "$1"
fi