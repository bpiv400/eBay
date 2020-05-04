#!/bin/bash
#$ -l m_mem_free=5G
#$ -N best_models
#$ -j y
#$ -o logs/sim/

python repo/sim/preprocessing/a_best_models.py