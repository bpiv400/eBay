#!/bin/bash
#$ -N chunk
#$ -j y
#$ -l m_mem_free=25G
#$ -m e -M 4102158912@vtext.com

scriptPath=repo/processing/chunk.py
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python $scriptPath --name $1
