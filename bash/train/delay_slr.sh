#!/bin/bash
#$ -N delay_slr
#$ -o logs/train/
#$ -j y

python repo/train/train_model.py --name delay_slr --smoothing