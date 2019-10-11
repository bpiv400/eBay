#!/bin/bash
#$ -t 1-16
#$ -l m_mem_free=60G
#$ -N cluster
#$ -j y
#$ -o logs/

python repo/simulator/train_cluster.py --id "$SGE_TASK_ID"
