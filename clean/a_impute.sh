#!/bin/bash
#$ -l m_mem_free=150G
#$ -N impute
#$ -j y
#$ -o logs/

python repo/clean/a_impute.py