#!/bin/bash
#$ -N prep_training 
#$ -js 1 
#$ -j y
#$ -l m_mem_free=15G

while getopts 'l:h:s:n:' flag; do
  case "${flag}" in
    l) low="${OPTARG}" ;;
    h) high="${OPTARG}" ;;
    s) step="${OPTARG}" ;;
    n) name="${OPTARG}" ;;
  esac
done

cd ~/eBay/data/$name
scriptPath=repo/trans_probs/mvp/prep.py  
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python "$scriptPath" --name $name-"$SGE_TASK_ID"_feats2.csv --dir $name --step $step --low $low --high $high 

