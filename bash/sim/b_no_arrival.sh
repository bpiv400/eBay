#!/bin/bash
#$ -l m_mem_free=50G
#$ -N no_arrival
#$ -j y
#$ -o logs/sim/

python repo/sim/preprocessing/b_no_arrival.py --part "$1"