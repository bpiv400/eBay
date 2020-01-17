#!/bin/bash
#$ -N delay_slr
#$ -o logs/train/
#$ -j y

python repo/train/smoothing.py --name delay_slr