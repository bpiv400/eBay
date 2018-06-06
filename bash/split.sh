#!/bin/bash
#$ -N JOBNAME
#$ -j y
#$ -l m_mem_free=12G
#$ -m e -M 4102158912@vtext.com

scriptPath = repo/processing/split.py
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python $scriptPath
