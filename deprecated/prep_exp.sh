#!/bin/bash

# first argument gives experiment name
# second argument gives prep type
cd ~/eBay
turns=( "b0" "b1" "b2" )
for j in "${turns[@]}"; 
do
    qsub -t 1-7 repo/bash/mvp_prep.sh -e $1 -p $2 -n toy -t $j 
    qsub -t 1-73 repo/bash/mvp_prep.sh -e $1 -p $2 -n train -t $j 
    qsub -t 1-18 repo/bash/mvp_prep.sh -e $1 -p $2 -n "test" -t $j
done


