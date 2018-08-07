#!/bin/bash
#$ -N make_seqs
#$ -j y
#$ -l m_mem_free=80G
#$ -m e -M 4102158912@vtext.com
#$ -js 1

# parse arguments
while getopts 'e:d:b' flag; do
  case "${flag}" in
    e) exp="${OPTARG}" ;;
    d) data="${OPTARG}" ;;
    b) b3="True" ;; #gives whether b3 should be used in training
  esac
done
echo $data
echo $exp
scriptPath="repo/rnn/make_seqs.py"
# separation and concatenation flags should be set in experiment name
types=( "toy" "train" "test" )
for type in "${types[@]}"
do
    if [ -z ${b3} ]; then
        echo "no b3"
        python $scriptPath --data $data --name $type --exp $exp
    else
        echo "yes b3"
        python $scriptPath --data $data --name $type --exp $exp --b3
    fi
done

