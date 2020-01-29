#!/bin/bash
#$ -t 1-9
#$ -N train
#$ -o logs/train/
#$ -j y

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/train/train_model.py --name arrival
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/train/train_model.py --name hist
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/train/train_model.py --name delay_byr
elif [ "$SGE_TASK_ID" == 4 ]
then
	python repo/train/train_model.py --name delay_slr
elif [ "$SGE_TASK_ID" == 5 ]
then
	python repo/train/train_model.py --name con_byr
elif [ "$SGE_TASK_ID" == 6 ]
then
	python repo/train/train_model.py --name con_slr
elif [ "$SGE_TASK_ID" == 7 ]
then
	python repo/train/train_model.py --name msg_byr
elif [ "$SGE_TASK_ID" == 8 ]
then
	python repo/train/train_model.py --name msg_slr
elif [ "$SGE_TASK_ID" == 9 ]
then
	python repo/train/train_model.py --name con_byr_no7
fi