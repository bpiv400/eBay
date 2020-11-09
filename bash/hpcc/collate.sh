#!/bin/bash
#$ -l m_mem_free=25G
#$ -N collate
#$ -j y
#$ -o logs/

ulimit -n 4096
python repo/sim/collate.py --part "$1" --type "$2"