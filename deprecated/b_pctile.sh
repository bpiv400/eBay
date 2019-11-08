#!/bin/bash
#$ -l m_mem_free=150G
#$ -N pctile
#$ -j y
#$ -o logs/

python repo/clean/b_pctile.py