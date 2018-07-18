#!/bin/bash

while getopts 'e:s:b:w:h:d:v:' flag; do
  case "${flag}" in
    e) exp_name="${OPTARG}" ;;
    s) batch_size="${OPTARG}" ;;
    b) batches="${OPTARG}" ;;
    w) wait_id="${OPTARG}" ;;
    h) hist_len="${OPTARG}" ;;
    d) dur_valid="${OPTARG}" ;;
    v) val_size="${OPTARG}" ;;
  esac
done
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay

# set default values
if [ -z "${val_size}" ]; then
    val_size=.05 ;
fi

if [ -z "${batch_size}" ]; then
    batch_size=32 ;
fi

if [ -z "${hist_len}" ]; then
    hist_len=1 ;
fi

if [ -z "${dur_valid}" ]; then
    dur_valid=1000 ;
fi

turns=( "b0" "b1" "b2" )
for t in "${turns[@]}"
do
  if [ -z "${wait_id}" ]; then
    echo "Curr Turn" $t
    qsub -js 1 repo/bash/train_model.sh -t $t -b $batches -e $exp_name -s $batch_size -h $hist_len -d $dur_valid -v $val_size;
  else
    qsub -js 1 -hold_jid $wait_id repo/bash/train_model.sh -t $t -b $batches -s $batch_size -e $exp_name -h $hist_len -d $dur_valid -v $val_size;
  fi
done


