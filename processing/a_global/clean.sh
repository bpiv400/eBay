#!/bin/bash
#$ -l m_mem_free=75G
#$ -N clean
#$ -j y
#$ -o logs/

python repo/processing/a_global/clean.py