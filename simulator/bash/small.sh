#!/bin/bash
#$ -t 1-8
#$ -N small
#$ -o logs/
#$ -j y

python repo/simulator/small.py --num "$SGE_TASK_ID"