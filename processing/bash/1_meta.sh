#!/bin/bash
#$ -l m_mem_free=100G
#$ -N meta
#$ -j y
#$ -o logs/

python repo/processing/1_meta.py