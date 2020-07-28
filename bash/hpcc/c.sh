#!/bin/bash
#$ -t 1-3
#$ -l m_mem_free=75G
#$ -N frames
#$ -j y
#$ -o logs/processing/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/processing/c_frames/lstg.py --part "$1"
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/processing/c_frames/thread.py --part "$1"
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/processing/c_frames/offer.py --part "$1"
fi