#!/bin/bash
#$ -t 1-2
#$ -l m_mem_free=75G
#$ -N chunk
#$ -j y
#$ -o logs/

python repo/processing/a_chunk/chunk.py