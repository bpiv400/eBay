#!/bin/bash
#$ -l m_mem_free=100G
#$ -N recombine
#$ -j y
#$ -o logs/$JOB_NAME-$JOB_ID.log

scriptPath=./repo/processing/3_recombine.py
python "$scriptPath"