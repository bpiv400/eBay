#!/bin/bash
#$ -N train_model 
#$ -js 1 
#$ -j y
#$ -l m_mem_free=40G
#$ -m e -M 4102158912@vtext.com

while getopts 'e:t:b:s:h:d:v:' flag; do
  case "${flag}" in    
    e) exp_name="${OPTARG}" ;;
    t) turn="${OPTARG}" ;;
    b) batches="${OPTARG}" ;;
    s) batch_size="${OPTARG}" ;;
    h) hist_len="${OPTARG}" ;;
    d) dur_valid="${OPTARG}" ;;
    v) val_size="${OPTARG}" ;;
  esac
done
if [ -z "$batch_size" ]; then
    batch_size=32 ;
fi
cd ~/eBay/data/
scriptPath=repo/trans_probs/mvp/train_model.py  
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python "$scriptPath" --batch_size $batch_size --batches $batches --exp $exp_name --turn $turn --hist_len $hist_len --dur_valid $dur_valid --valid_size $val_size
