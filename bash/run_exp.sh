#!/bin/bash

while getopts 'e:f:b:w:' flag; do
  case "${flag}" in
    e) exp_name="${OPTARG}" ;;
    f) file="${OPTARG}" ;;
    b) batches="${OPTARG}" ;;
    w) wait_id="${OPTARG}" ;;
  esac
done
scriptPath=repo/trans_probs/mvp/$file.py
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
turns=( "b0" "b1" "b2" )
for t in "${turns[@]}"
do
  if [-z ${wait_id+x} ]; then
    echo "Curr Turn" $t
    qsub -js 1 repo/bash/mvp_train.sh -t $t -b $batches -f $file -e $exp_name ;
  else
    qsub -js 1 -hold_jid $wait_id repo/bash/mvp_train.sh -t $t -b $batches -f $file -e $exp_name ;
  fi
done
