#!/bin/bash
#$ -N prep_rnn
#$ -js 1
#$ -j y
#$ -l m_mem_free=15G
# prep gives the type of prep to use, corresponding to
# repo/trans_probs/mvp/prep$prep.py files
# exp gives the name of the experiment being conducted
# name gives the type of the file being used
# (toy, train, etc)

while getopts 'e:n:p:' flag; do
  case "${flag}" in
    e) exp="${OPTARG}" ;;
    n) name="${OPTARG}" ;;
    p) prep="${OPTARG}" ;;
  esac
done
cd ~/eBay/data/$name
scriptPath=repo/rnn/prep_rnn$prep.py
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python "$scriptPath" --name $name-"$SGE_TASK_ID"_feats2.csv --dir $name --exp $exp