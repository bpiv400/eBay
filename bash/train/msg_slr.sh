#!/bin/bash
#$ -N msg_slr
#$ -o logs/train/
#$ -j y

python repo/models/train.py --name msg_slr