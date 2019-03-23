#!/bin/bash
#$ -l m_mem_free=10G
#$ -N lda
#$ -j y
#$ -o logs/

python repo/processing/0_lda.py