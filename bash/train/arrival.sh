#!/bin/bash
#$ -N arrival
#$ -o logs/train/
#$ -j y

python repo/train/train_model.py --name arrival