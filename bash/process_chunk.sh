#!/bin/bash
#$ -m e -M 4102158912@vtext.com

scriptPath=repo/processing/2_process_chunk.py
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python "$scriptPath" --num "$SGE_TASK_ID"
