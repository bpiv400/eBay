#!/bin/bash
#$ -N delay_byr
#$ -o logs/train/
#$ -j y

python repo/train/train_model.py --name delay_byr