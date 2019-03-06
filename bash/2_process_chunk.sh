#!/bin/bash
#$ -t 1-256
#$ -N process_chunk
#$ -j y
#$ -o logs/$JOB_NAME-$JOB_ID-$SGE_TASK_ID.log

scriptPath=repo/processing/2_process_chunk.py
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python "$scriptPath" --num "$SGE_TASK_ID"
