#!/bin/bash
#$ -m e -M 4102158912@vtext.com
#$ -l m_mem_free=150G
#$ -N recombine
#$ -j y
#$ -o logs/$JOB_NAME-$JOB_ID.log

scriptPath=repo/processing/3_recombine.py
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python "$scriptPath"