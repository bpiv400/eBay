#!/bin/bash
#$ -N arrival
#$ -o logs/train/
#$ -j y

python repo/models/train.py --name arrival