#!/bin/bash
#$ -N msg_byr
#$ -o logs/train/
#$ -j y

python repo/models/train.py --name msg_byr