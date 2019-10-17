#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=10G
#$ -t 1-28
#$ -N delay_byr
#$ -o logs/
#$ -j y

python repo/simulator/train.py --model delay_byr --id "$SGE_TASK_ID"