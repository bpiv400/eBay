#!/bin/bash
#$ -t 1-48
#$ -N delay_byr
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model delay_byr --id "$SGE_TASK_ID"