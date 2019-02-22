#!/bin/bash
#$ -j y
#$ -l m_mem_free=50G
#$ -m e -M 4153141889@vtext.com
#$ -o logs/

python "$1"