#!/bin/bash
#$ -t 1-256
#$ -q short.q
#$ -N process_chunk
#$ -j y
#$ -o logs/

scriptPath=repo/processing/2_process_chunk.py
python "$scriptPath" --num "$SGE_TASK_ID"
