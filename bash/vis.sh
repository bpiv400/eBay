#!/bin/bash
#$ -N vis 
#$ -j y
#$ -l m_mem_free=24G
#$ -m e -M 4102158912@vtext.com

scriptPath=repo/processing/visualization.py
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python $scriptPath
