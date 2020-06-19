#!/bin/bash
#$ -l m_mem_free=25G
#$ -N concat_outcomes
#$ -j y
#$ -o logs/sim/

python repo/sim/outcomes/concat.py --part "$1"