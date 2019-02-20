#!/bin/bash
#$ -N norm
#$ -js 1
#$ -j y
#$ -l m_mem_free=100G
#$ -m e -M 4102158912@vtext.com
while getopts 'e:p:' flag; do
  case "${flag}" in
    e) exp="${OPTARG}" ;;
    p) prep="${OPTARG}" ;;
  esac
done

scriptPath=repo/rnn/norm_rnn$prep.py
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
types=( "train" "toy" "test" )
for k in "${types[@]}"
do
    python "$scriptPath" --name $k --exp $exp
done
