#!/bin/bash
#$ -N delay_slr
#$ -o logs/train/
#$ -j y

python repo/models/train.py --name delay_slr