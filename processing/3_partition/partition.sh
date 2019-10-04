#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=50G
#$ -N partition
#$ -j y
#$ -o logs/

python repo/processing/3_partition/partition.py