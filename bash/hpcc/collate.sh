#!/bin/bash
#$ -l m_mem_free=25G
#$ -N collate
#$ -j y
#$ -o logs/collate/

ulimit -n 4096
if [ "$2" == "values" ]; then
  python repo/sim/collate.py --part "$1" --values
else
  python repo/sim/collate.py --part "$1"
fi