#!/bin/bash
#$ -l m_mem_free=50G
#$ -N chunk
#$ -j y
#$ -o logs/$JOB_NAME-$JOB_ID.log

scriptPath=repo/processing/1_chunk.py
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python "$scriptPath"