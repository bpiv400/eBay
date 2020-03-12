#!/bin/bash
#$ -t 1-7
#$ -N con
#$ -o logs/train/
#$ -j y

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/train/train_model.py --name con1 --dropout
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/train/train_model.py --name con2 --dropout
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/train/train_model.py --name con3 --dropout
elif [ "$SGE_TASK_ID" == 4 ]
then
	python repo/train/train_model.py --name con4 --dropout
elif [ "$SGE_TASK_ID" == 5 ]
then
	python repo/train/train_model.py --name con5 --dropout
elif [ "$SGE_TASK_ID" == 6 ]
then
	python repo/train/train_model.py --name con6 --dropout
elif [ "$SGE_TASK_ID" == 7 ]
then
	python repo/train/train_model.py --name con7 --dropout
fi