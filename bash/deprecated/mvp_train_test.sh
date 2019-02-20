#!/bin/bash
#$ -N mvp_train 
#$ -js 1 
#$ -j y
#$ -l m_mem_free=18G
#$ -m e -M 4102158912@vtext.com

while getopts 'e:t:s:b:' flag; do
  case "${flag}" in    
    e) exp_name="${OPTARG}" ;;
    t) turn="${OPTARG}" ;;
    s) batch_size="${OPTARG}" ;; 
    b) batches="${OPTARG}" ;;
  esac
done
cd ~/eBay/data/
scriptPath=repo/trans_probs/mvp/train_model.py  
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python "$scriptPath" --batches $batches --batch_size $batch_size --exp $exp_name --turn $turn
