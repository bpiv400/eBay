#!/bin/bash
#$ -l m_mem_free=150G
#$ -N days
#$ -j y
#$ -o logs/

python repo/processing/5_inputs/days.py
