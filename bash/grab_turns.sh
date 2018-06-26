#!/bin/bash
#$ -N grab_turns 
#$ -js 1 
#$ -j y
#$ -l m_mem_free=15G

cd ~/eBay/data/$1
scriptPath=repo/processing/extract_turns.py  
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python "$scriptPath" --name $1-"$SGE_TASK_ID"_feats2.csv --dir $1 --turn 0
python "$scriptPath" --name $1-"$SGE_TASK_ID"_feats2.csv --dir $1 --turn 0 --seller 
python "$scriptPath" --name $1-"$SGE_TASK_ID"_feats2.csv --dir $1 --turn 1 
python "$scriptPath" --name $1-"$SGE_TASK_ID"_feats2.csv --dir $1 --turn 1 --seller 
python "$scriptPath" --name $1-"$SGE_TASK_ID"_feats2.csv --dir $1 --turn 2  

