#!/bin/bash
#$ -t 1-48
#$ -N delay_slr
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model delay_slr --id "$SGE_TASK_ID"