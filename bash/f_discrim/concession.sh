#!/bin/bash
#$ -t 1-7
#$ -l m_mem_free=50G
#$ -N f_con
#$ -j y
#$ -o logs/processing/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/processing/f_discrim/offer.py --part "$1" --outcome con --turn 1
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/processing/f_discrim/offer.py --part "$1" --outcome con --turn 2
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/processing/f_discrim/offer.py --part "$1" --outcome con --turn 3
elif [ "$SGE_TASK_ID" == 4 ]
then
	python repo/processing/f_discrim/offer.py --part "$1" --outcome con --turn 4
elif [ "$SGE_TASK_ID" == 5 ]
then
	python repo/processing/f_discrim/offer.py --part "$1" --outcome con --turn 5
elif [ "$SGE_TASK_ID" == 6 ]
then
	python repo/processing/f_discrim/offer.py --part "$1" --outcome con --turn 6
elif [ "$SGE_TASK_ID" == 7 ]
then
	python repo/processing/f_discrim/offer.py --part "$1" --outcome con --turn 7
fi