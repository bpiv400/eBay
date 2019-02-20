#!/bin/bash

# first argument gives experiment name
# second argument gives prep type
cd ~/eBay
echo $1
echo $2
qsub -t 1-7 repo/bash/prep_rnn.sh -e $1 -n toy -p $2
qsub -t 1-73 repo/bash/prep_rnn.sh -e $1 -n train -p $2
qsub -t 1-18 repo/bash/prep_rnn.sh -e $1 -n "test" -p $2



