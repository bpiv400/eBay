#!/bin/bash
#$ -N recover 
#$ -j y
#$ -l m_mem_free=15G
#$ -m e -M 4102158912@vtext.com
#$ -js 1

first=true
for i in $(find data/toy/ -maxdepth 1 -type f -name "toy-[0-9].csv"); 
do
    if [ "$first" = true ] ; then
        cp $i data/toy.csv
        first=false
    else
        tail -n +2 $i >> data/toy.csv
    fi
done


