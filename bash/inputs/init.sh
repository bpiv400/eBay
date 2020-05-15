#!/bin/bash
#$ -t 1-8
#$ -l m_mem_free=50G
#$ -N init
#$ -j y
#$ -o logs/inputs/

if [ "$SGE_TASK_ID" == 1 ]
then
	python repo/inputs/init_policy.py --part "$1" --role slr
elif [ "$SGE_TASK_ID" == 2 ]
then
	python repo/inputs/init_value.py --part "$1" --role slr
elif [ "$SGE_TASK_ID" == 3 ]
then
	python repo/inputs/init_policy.py --part "$1" --role slr --delay
elif [ "$SGE_TASK_ID" == 4 ]
then
	python repo/inputs/init_value.py --part "$1" --role slr --delay
elif [ "$SGE_TASK_ID" == 5 ]
then
	python repo/inputs/init_policy.py --part "$1" --role byr
elif [ "$SGE_TASK_ID" == 6 ]
then
	python repo/inputs/init_value.py --part "$1" --role byr
elif [ "$SGE_TASK_ID" == 7 ]
then
	python repo/inputs/init_policy.py --part "$1" --role byr --delay
elif [ "$SGE_TASK_ID" == 8 ]
then
	python repo/inputs/init_value.py --part "$1" --role byr --delay
fi