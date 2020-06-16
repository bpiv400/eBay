#!/bin/bash
#$ -l m_mem_free=25G
#$ -N post
#$ -j y
#$ -o logs/train/

python repo/train/post.py