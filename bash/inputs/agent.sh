#!/bin/bash
#$ -l m_mem_free=50G
#$ -N agent
#$ -j y
#$ -o logs/inputs/

python repo/inputs/agent.py
