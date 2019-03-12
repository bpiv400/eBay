#!/bin/bash
#$ -l m_mem_free=150G
#$ -N chunk
#$ -j y
#$ -o logs/$JOB_NAME-$JOB_ID.log

scriptPath=./repo/processing/1_chunk.py
python "$scriptPath"