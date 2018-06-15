#!/bin/bash
#$ -N first_features_debug
#$ -js 1 
#$ -j y
#$ -l m_mem_free=10G
#$ -m e -M 4102158912@vtext.com
cd ~/eBay/data/$1
scriptPath=repo/processing/first_pass_features.py  
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python "$scriptPath" --name $1-1.csv --dir $1
