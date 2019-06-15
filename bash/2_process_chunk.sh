#!/bin/bash
#$ -t 1-256
#$ -q short.q
#$ -N process_chunk
#$ -j y
#$ -o logs/

python repo/processing/2_process_chunk.py --num "$SGE_TASK_ID"