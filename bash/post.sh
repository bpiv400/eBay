#!/bin/bash
#$ -l m_mem_free=75G
#$ -N post
#$ -j y
#$ -o logs/train/

python repo/train/post.py --set "$1"