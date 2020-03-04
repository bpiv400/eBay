#!/bin/bash
#$ -t 1-2
#$ -N init
#$ -o logs/train/
#$ -j y

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/train/train_model.py --name init_slr
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/train/train_model.py --name init_byr
fi