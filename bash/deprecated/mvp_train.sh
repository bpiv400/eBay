#!/bin/bash
#$ -N mvp_train 
#$ -js 1 
#$ -j y
#$ -l m_mem_free=18G
#$ -m e -M 4102158912@vtext.com

while getopts 'e:t:f:b:' flag; do
  case "${flag}" in    
    e) exp_name="${OPTARG}" ;;
    t) turn="${OPTARG}" ;;
    f) file="${OPTARG}" ;;
    b) batches="${OPTARG}" ;;
  esac
done
cd ~/eBay/data/
scriptPath=repo/trans_probs/mvp/$file.py  
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python "$scriptPath" --name train --batches $batches --exp $exp_name --turn $turn
