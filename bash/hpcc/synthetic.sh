#!/bin/bash
#$ -l m_mem_free=25G
#$ -N synthetic
#$ -j y
#$ -o logs/

ulimit -n 4096
python repo/sim/synthetic.py --part "$1"