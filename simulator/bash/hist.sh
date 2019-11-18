#!/bin/bash
#$ -l m_mem_free=30G
#$ -t 1-144
#$ -N hist
#$ -o logs/
#$ -j y

python weka/eBay/repo/simulator/train.py --model hist --id "$SGE_TASK_ID"