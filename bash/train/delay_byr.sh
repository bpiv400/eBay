#!/bin/bash
#$ -N delay_byr
#$ -o logs/train/
#$ -j y

python repo/models/train.py --name delay_byr