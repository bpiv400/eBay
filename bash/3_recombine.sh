#!/bin/bash
#$ -l m_mem_free=50G
#$ -N recombine
#$ -j y
#$ -o logs/

python repo/processing/3_recombine.py