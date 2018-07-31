#!/bin/bash
#$ -N concat
#$ -j y
#$ -l m_mem_free=15G
#$ -m e -M 4102158912@vtext.com
#$ -js 1

# first argument gives experiment name
# second argument gives file type
types=( "toy" "test" "train" )
for k in "${types[@]}"
do
    echo data/exps/$1/"$k"_concat.csv
    if [ -f data/exps/$1/"$k"_concat.csv ] ; then
        rm data/exps/$1/"$k"_concat.csv ;
    fi
    first=true
    for i in $(find data/exps/$1 -type f -name "*$k*");
    do
        echo $i
        echo $first
        if [ "$first" = true ] ; then
            cp $i data/exps/$1/"$k"_concat.csv
            first=false ;
        else
            # echo "tail" ;
            tail -n +2 $i >> data/exps/$1/"$k"_concat.csv ;
        fi
    done
done


