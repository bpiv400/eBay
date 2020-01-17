#!/bin/bash
#$ -N arrival
#$ -o logs/train/
#$ -j y

python repo/train/smoothing.py --name arrival