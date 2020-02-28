#!/bin/bash
#$ -t 1-6
#$ -N msg
#$ -o logs/train/
#$ -j y

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/train/train_model.py --name msg1
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/train/train_model.py --name msg2
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/train/train_model.py --name msg3
elif [ "$SGE_TASK_ID" == 4 ]
then
	python repo/train/train_model.py --name msg4
elif [ "$SGE_TASK_ID" == 5 ]
then
	python repo/train/train_model.py --name msg5
elif [ "$SGE_TASK_ID" == 6 ]
then
	python repo/train/train_model.py --name msg6
fi