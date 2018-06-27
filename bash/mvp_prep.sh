#!/bin/bash
#$ -N prep_training 
#$ -js 1 
#$ -j y
#$ -l m_mem_free=15G

while getopts 'l:h:s:n:t:' flag; do
  case "${flag}" in
    l) low="${OPTARG}" ;;
    h) high="${OPTARG}" ;;
    s) step="${OPTARG}" ;;
    n) name="${OPTARG}" ;;
    t) turn="${OPTARG}" ;;
  esac
done
echo $turn
echo $name
cd ~/eBay/data/$name
scriptPath=repo/trans_probs/mvp/prep.py  
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python "$scriptPath" --name $name-"$SGE_TASK_ID".csv --dir $name --step $step --low $low --high $high --turn $turn 
