#!/bin/bash
#$ -t 1-512
#$ -N process_chunk
#$ -j y
#$ -m e -M 4102158912@vtext.com
#$ -o logs/$JOB_NAME-$JOB_ID-$SGE_TASK_ID.log
scriptPath=repo/processing/2_process_chunk.py
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python "$scriptPath" --num "$SGE_TASK_ID"
