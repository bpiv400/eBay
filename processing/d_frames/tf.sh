#!/bin/bash
#$ -q all.q
#$ -t 1-3
#$ -l m_mem_free=50G
#$ -N frames_tf
#$ -j y
#$ -o logs/

python repo/processing/d_frames/tf.py --num "$SGE_TASK_ID"