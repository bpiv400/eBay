#!/bin/bash
#$ -l m_mem_free=30G
#$ -t 1-144
#$ -N delay_slr
#$ -o logs/
#$ -j y

python weka/eBay/repo/simulator/train.py --model delay_slr --id "$SGE_TASK_ID"