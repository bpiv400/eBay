#!/bin/bash
#$ -l m_mem_free=150G
#$ -N pctile
#$ -j y
#$ -o logs/

python repo/processing/clean/b_pctile.py