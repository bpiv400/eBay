#!/bin/bash
#$ -N norm
#$ -js 1
#$ -j y
#$ -l m_mem_free=50G
#$ -m e -M 4102158912@vtext.com
while getopts 'e:' flag; do
  case "${flag}" in
    e) exp="${OPTARG}" ;;
  esac
done

cd ~/eBay/data/$name
scriptPath=repo/trans_probs/mvp/norm.py
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
types=( "train" "toy" "test" )
turns=( "b0" "b1" "b2" )
for k in "${types[@]}"
do
    for j in "${turns[@]}"
    do
        python "$scriptPath" --name $k --exp $exp --turn $j
    done
done
