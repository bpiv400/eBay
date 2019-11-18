#!/bin/bash
#$ -l m_mem_free=30G
#$ -t 1-144
#$ -N msg_byr
#$ -o ~/logs/
#$ -j y

python ~/weka/eBay/repo/simulator/train.py --model msg_byr --id "$SGE_TASK_ID"