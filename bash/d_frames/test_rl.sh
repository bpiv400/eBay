#!/bin/bash
#$ -t 1-5
#$ -l m_mem_free=75G
#$ -N d_test_rl
#$ -j y
#$ -o logs/processing/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/processing/d_frames/lstg.py --part test_rl
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/processing/d_frames/thread.py --part test_rl
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/processing/d_frames/arrival.py --part test_rl
elif [ "$SGE_TASK_ID" == 4 ]
then
	python repo/processing/d_frames/delay.py --part test_rl
elif [ "$SGE_TASK_ID" == 5 ]
then
	python repo/processing/d_frames/offer.py --part test_rl
fi