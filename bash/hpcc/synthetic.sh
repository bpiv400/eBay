#!/bin/bash
#$ -l m_mem_free=25G
#$ -N synthetic
#$ -j y
#$ -o logs/collate/

ulimit -n 4096
python repo/sim/synthetic.py