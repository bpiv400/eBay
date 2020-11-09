#!/bin/bash
#$ -l m_mem_free=25G
#$ -N agent
#$ -j y
#$ -o logs/collate/

ulimit -n 4096
python repo/agent/eval/collate.py "$1"