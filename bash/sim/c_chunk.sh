#!/bin/bash
#$ -l m_mem_free=50G
#$ -N chunk
#$ -j y
#$ -o logs/sim/

python repo/sim/preprocessing/c_chunk.py --part "$1"
