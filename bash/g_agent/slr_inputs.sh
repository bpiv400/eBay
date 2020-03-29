#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=50G
#$ -N seller_train_inputs
#$ -j y
#$ -o logs/

python repo/agent/a_inputs/seller_inputs.py --part train_rl
