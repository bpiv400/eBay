#!/bin/bash
#$ -t 1-26
#$ -pe openmp 4
#$ -N train
#$ -o logs/train/
#$ -j y

python repo/train/train_model.py --name "$1" --dropout "$SGE_TASK_ID"