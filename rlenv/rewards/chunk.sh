#!/bin/bash
#$ -q all.q
#$ -l m_mem_free=128G
#$ -N rewards_chunk
#$ -j y
#$ -o logs/

python repo/rlenv/rewards/chunk.py --part train_rl