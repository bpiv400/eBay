#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=50G
#$ -N partition
#$ -j y
#$ -o logs/

python repo/processing/c_partition/partition.py