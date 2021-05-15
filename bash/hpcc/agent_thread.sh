#!/bin/bash
#$ -l m_mem_free=25G
#$ -N agent_thread
#$ -j y
#$ -o logs/collate/

ulimit -n 4096
python repo/agent_thread.py --part "$1"