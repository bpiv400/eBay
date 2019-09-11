#!/bin/bash
#$ -N test_rnn 
#$ -j y
#$ -l m_mem_free=15G
#$ -m e -M 4102158912@vtext.com

scriptPath=repo/processing/prep_rnn.py
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python $scriptPath --file toy-1_feats2.csv --dir toy
