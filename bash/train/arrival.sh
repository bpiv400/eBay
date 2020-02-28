#!/bin/bash
#$ -t 1-3
#$ -N arrival
#$ -o logs/train/
#$ -j y

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/train/train_model.py --name first_arrival
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/train/train_model.py --name next_arrival
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/train/train_model.py --name hist
fi