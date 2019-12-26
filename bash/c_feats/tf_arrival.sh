#!/bin/bash
#$ -t 1-281
#$ -q short.q
#$ -l m_mem_free=10G
#$ -N c_tf_arrival
#$ -j y
#$ -o logs/processing/

python repo/processing/c_feats/tf.py --num "$SGE_TASK_ID" --arrival
