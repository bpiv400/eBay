#!/bin/bash
#$ -N delay_byr
#$ -o logs/train/
#$ -j y

python repo/model/train.py --name delay_byr