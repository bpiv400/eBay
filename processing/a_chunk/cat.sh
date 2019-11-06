#!/bin/bash
#$ -l m_mem_free=100G
#$ -N chunk_cat
#$ -j y
#$ -o logs/

python repo/processing/a_chunk/cat.py