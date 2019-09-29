#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=150G
#$ -N feats
#$ -j y
#$ -o logs/

python repo/processing/3_recombine/feats.py