#!/bin/bash
#$ -N msg_slr
#$ -o logs/train/
#$ -j y

python repo/train/train_model.py --name msg_slr