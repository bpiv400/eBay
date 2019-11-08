#!/bin/bash
#$ -l m_mem_free=100G
#$ -N chunk_slr
#$ -j y
#$ -o logs/

python repo/processing/a_chunk/slr.py