#!/bin/bash
#$ -j y
#$ -l m_mem_free=20G
#$ -m e -M etangreen@gmail.com
#$ -o logs/

cd ~/Dropbox/eBay/data/
stata do "$1"