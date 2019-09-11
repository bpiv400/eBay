#!/bin/bash
#$ -N prep_training 
#$ -js 1 
#$ -j y
#$ -l m_mem_free=15G
# prep gives the type of prep to use, corresponding to
# repo/trans_probs/mvp/prep$prep.py files

# exp gives the name of the experiment being conducted
#
# turn gives the turn for which the experiment is 
# currently being conducted
# 
# name gives the type of the file being used
# (toy, train, etc)

while getopts 'e:n:t:p:' flag; do
  case "${flag}" in
    e) exp="${OPTARG}" ;;
    n) name="${OPTARG}" ;;
    t) turn="${OPTARG}" ;;
    p) prep="${OPTARG}" ;;
  esac
done
echo $turn
echo $name
cd ~/eBay/data/$name
scriptPath=repo/trans_probs/mvp/prep$prep.py  
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python "$scriptPath" --name $name-"$SGE_TASK_ID".csv --dir $name --turn $turn --exp $exp 
