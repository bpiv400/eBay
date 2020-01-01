#!/bin/bash
#$ -N arrival
#$ -o logs/train/
#$ -j y

python repo/model/train.py --name arrival