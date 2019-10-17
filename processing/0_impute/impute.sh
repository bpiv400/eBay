#!/bin/bash
#$ -l m_mem_free=100G
#$ -N impute
#$ -j y
#$ -o logs/

python repo/processing/0_impute/impute.py