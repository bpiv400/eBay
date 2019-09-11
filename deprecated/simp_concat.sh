#!/bin/bash
#$ -N concat
#$ -j y
#$ -l m_mem_free=15G
#$ -m e -M 4102158912@vtext.com
#$ -js 1
first=true
for i in $(find data/$1/turns/$2/$j -type f -name "*$1*"); 
do
    if [ "$first" = true ] ; then
        cp $i data/$1/turns/$2/"$1"_concat.csv
        first=false ;
    else
        # echo "tail" ;
        tail -n +2 $i >> data/$1/turns/$2/"$1"_concat.csv ;
    fi
done



