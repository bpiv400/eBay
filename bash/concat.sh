#!/bin/bash
#$ -N concat
#$ -j y
#$ -l m_mem_free=15G
#$ -m e -M 4102158912@vtext.com
#$ -js 1

first=true
for i in $(find data/$1 -type f -name "*feats2.csv"); do
    if [ "$first" = true ] ; then
        cp $i data/$1/$1_concat.csv
        first=false
    else
        tail -n +2 $i >> data/$1/$1_concat.csv
    fi
done
