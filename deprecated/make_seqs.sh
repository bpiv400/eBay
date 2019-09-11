#!/bin/bash
#$ -N make_seqs
#$ -j y
#$ -l m_mem_free=80G
#$ -m e -M 4102158912@vtext.com
#$ -js 1

# parse arguments
while getopts 'd:bi' flag; do
  case "${flag}" in
    d) data="${OPTARG}" ;;
    b) b3="True" ;; #gives whether b3 should be used in training
    i) inds="True" ;; # gives whether indicators should be created for the sequence input number
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
        if [ -z ${inds} ]; then
            echo "no inds"
            python $scriptPath --data $data --name $type
        else
            echo "yes inds"
            python $scriptPath --data $data --name $type --inds
        fi
    else
        echo "yes b3"
        if [ -z ${inds} ]; then
            echo "no inds"
            python $scriptPath --data $data --name $type --b3
        else
            echo "yes inds"
            python $scriptPath --data $data --name $type --inds --b3
        fi
    fi
done

