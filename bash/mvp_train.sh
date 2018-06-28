#!/bin/bash
#$ -N mvp_train 
#$ -js 1 
#$ -j y
#$ -l m_mem_free=50G

while getopts 'n:e:t:' flag; do
  case "${flag}" in    
    n) name="${OPTARG}" ;;
    e) exp_name="${OPTARG}" ;;
    t) turn="${OPTARG}" ;;
  esac
done
cd ~/eBay/data/$name
scriptPath=repo/trans_probs/mvp/train_nn.py  
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python "$scriptPath" --name $name --exp $exp_name --turn $turn 
