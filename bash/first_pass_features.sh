#!/bin/bash
#$ -N first_features 
#$ -j y
#$ -l m_mem_free=50G
#$ -m e -M 4102158912@vtext.com

scriptPath="repo/processing/first_pass_features.py data/toy.csv"  
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python "$scriptPath"
