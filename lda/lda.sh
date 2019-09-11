#!/bin/bash
#$ -t 2-20
#$ -q short.q
#$ -N lda
#$ -j y
#$ -o logs/

python repo/lda/train.py --topics "$SGE_TASK_ID"