#!/bin/bash
#$ -N delay_byr
#$ -o logs/train/
#$ -j y

python repo/train/smoothing.py --name delay_byr