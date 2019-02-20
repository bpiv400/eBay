#!/bin/bash
#$ -N JOBNAME
#$ -j y
#$ -l m_mem_free=10G
#$ -m e -M 4102158912@vtext.com
#$ -js 1

scriptPath=repo/test/mem_test.py
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python "$scriptPath"
