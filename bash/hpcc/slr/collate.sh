#!/bin/bash
#$ -l m_mem_free=25G
#$ -N slr_collate
#$ -j y
#$ -o logs/collate/

ulimit -n 4096
python repo/agent/eval/collate.py --delta "$1"