#!/bin/bash
#$ -N concat
#$ -j y
#$ -l m_mem_free=15G
#$ -m e -M 4102158912@vtext.com
#$ -js 1


turns=( "b0" "b1" "b2" "s0" "s1" )
for j in "${turns[@]}" 
do
    first=true
    for i in $(find data/curr_exp/$j -type f -name "*$1*"); 
    do
        if [ "$first" = true ] ; then
            cp $i data/curr_exp/$j/$1_concat.csv
            first=false
        else
            tail -n +2 $i >> data/curr_exp/$j/$1_concat.csv
        fi
    done
done


