#!/bin/bash
#$ -N concat
#$ -j y
#$ -l m_mem_free=15G
#$ -m e -M 4102158912@vtext.com
#$ -js 1

# first argument gives experiment name
# second argument gives file type 
# (toy, test, train, etc)
turns=( "b0" "b1" "b2" )
for j in "${turns[@]}" 
do   
    first=true
    for i in $(find data/exps/$1/$j -type f -name "*$2*"); 
    do
        echo $j
        echo $first
        echo $i
        if [ "$first" = true ] ; then
            cp $i data/exps/$1/$j/$2_concat.csv
            first=false ;
        else
            # echo "tail" ;
            tail -n +2 $i >> data/exps/$1/$j/$2_concat.csv ;
        fi
    done
done


