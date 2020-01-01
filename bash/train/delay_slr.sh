#!/bin/bash
#$ -N delay_slr
#$ -o logs/train/
#$ -j y

python repo/model/train.py --name delay_slr