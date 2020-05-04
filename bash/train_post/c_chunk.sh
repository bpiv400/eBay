#!/bin/bash
#$ -l m_mem_free=50G
#$ -N chunk
#$ -j y
#$ -o logs/train/

python repo/train/post/c_chunk.py --part "$1"
