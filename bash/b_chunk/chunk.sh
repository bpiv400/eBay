#!/bin/bash
#$ -t 1-2
#$ -l m_mem_free=75G
#$ -N chunk
#$ -j y
#$ -o logs/

python repo/processing/b_chunk/chunk.py --num "$SGE_TASK_ID"