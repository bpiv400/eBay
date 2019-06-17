#!/bin/bash
#$ -t 1-256
#$ -q short.q
#$ -N events
#$ -j y
#$ -o logs/

python repo/processing/2_events_lstgs.py --num "$SGE_TASK_ID"