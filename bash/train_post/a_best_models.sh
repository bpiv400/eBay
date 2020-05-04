#!/bin/bash
#$ -l m_mem_free=5G
#$ -N best_models
#$ -j y
#$ -o logs/train/

python repo/train_post/a_best_models.py