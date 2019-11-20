#!/bin/bash
#$ -t 1-512
#$ -q short.q
#$ -l m_mem_free=8G
#$ -N reward_test
#$ -j n
#$ -o logs/
#$ -e error/

 python repo/rlenv/rewards/generate_rewards.py --num "$SGE_TASK_ID" --id 1 --part train_rl