#!/bin/bash
#$ -l m_mem_free=60G
#$ -t 1-1296
#$ -N delay_byr
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model delay_byr --id "$SGE_TASK_ID"