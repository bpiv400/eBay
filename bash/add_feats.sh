#!/bin/bash
#$ -N add_feats 
#$ -js 1 
#$ -j y
#$ -l m_mem_free=15G

cd ~/eBay/data/$1
scriptPath=repo/processing/add_feats.py  
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python "$scriptPath" --name $1-"$SGE_TASK_ID"_feats.csv --dir $1

