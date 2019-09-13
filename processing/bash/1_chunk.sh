#!/bin/bash
#$ -l m_mem_free=150G
#$ -N chunk
#$ -j y
#$ -o logs/

python repo/processing/1_chunk.py