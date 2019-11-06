#!/bin/bash
#$ -l m_mem_free=250G
#$ -N recombine
#$ -j y
#$ -o logs/

python repo/processing/4_recombine.py