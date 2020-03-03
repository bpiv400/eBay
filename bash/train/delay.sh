
#!/bin/bash
#$ -t 2-7
#$ -N delay
#$ -o logs/train/
#$ -j y

if [ "$SGE_TASK_ID" == 2 ]
then
	python repo/train/train_model.py --name delay2
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/train/train_model.py --name delay3
elif [ "$SGE_TASK_ID" == 4 ]
then
	python repo/train/train_model.py --name delay4
elif [ "$SGE_TASK_ID" == 5 ]
then
	python repo/train/train_model.py --name delay5
elif [ "$SGE_TASK_ID" == 6 ]
then
	python repo/train/train_model.py --name delay6
elif [ "$SGE_TASK_ID" == 7 ]
then
	python repo/train/train_model.py --name delay7
fi