#!/bin/bash

while getopts ":b:" opt; do
  case $opt in
    b) BYR_HIST="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

LAST=$(bash repo/bash/rl/last.sh)
printf "Running %d experiments\n" "$((LAST+1))"
for ((i=0; i<=LAST; i++))
do
  printf "Experiment #%d: " $i
  GPU=$(bash repo/bash/gpu.sh)
  printf "starting on GPU %d\n" "$GPU"
  python repo/agent/train.py --gpu "$GPU" --exp $i --log --byr_hist "$BYR_HIST" &>/dev/null &
  sleep 15
done