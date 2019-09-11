#!/bin/bash
#$ -N train_tensor_model 
#$ -js 1 
#$ -j y
#$ -l m_mem_free=60G
#$ -m e -M 4102158912@vtext.com

while getopts 'e:b:s:h:d:v:' flag; do
  case "${flag}" in
    e) exp_name="${OPTARG}" ;;
    b) batches="${OPTARG}" ;;
    s) batch_size="${OPTARG}" ;;
    h) hist_len="${OPTARG}" ;;
    d) dur_valid="${OPTARG}" ;;
    v) val_size="${OPTARG}" ;;
  esac
done

scriptPath=repo/rnn/train_tensor.py

if [ -z "${val_size}" ]; then
    val_size=.05 ;
fi

# if [ -z "${init}" ]; then
#     init=False ;
# fi
# 
# if [ -z "${zeros}" ]; then
#     zeros=False ;
# fi

if [ -z "${batch_size}" ]; then
    batch_size=32 ;
fi

if [ -z "${hist_len}" ]; then
    hist_len=10 ;
fi

if [ -z "${dur_valid}" ]; then
    dur_valid=1000 ;
fi

if [ -z "$batches" ]; then
    batches=10 ;
fi

cd ~/eBay/data/

cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay

python "$scriptPath" --batch_size $batch_size --batches $batches --exp $exp_name --hist_len $hist_len --dur_valid $dur_valid --valid_size $val_size
