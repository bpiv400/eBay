#!/bin/bash
#$ -q short.q
#$ -l m_mem_free=4G
#$ -N ram_test
#$ -j y
#$ -o logs/

python repo/rlenv/simulator/generate.py --part train_rl --num 1 --values
