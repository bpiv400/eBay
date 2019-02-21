#!/bin/bash
#$ -N chunk
#$ -j y
#$ -l m_mem_free=50G
#$ -m e -M 4153141889@vtext.com

python repo/processing/1_chunks.py
