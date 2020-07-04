#!/bin/bash

# wait until midnight
if [ "$1" == "wait" ]; then
  SECONDS=$(((24 * 3600) - $(date -d "1970-01-01 UTC $(date +%T)" +%s)))
  printf "Waiting %d seconds until midnight" $SECONDS
  sleep $SECONDS
fi

# index of last experiment
LAST=$(tail -n1 "data/agent/logs/exps.csv" | cut -d, -f1)
printf "Running %d experiments\n" "$LAST"

for ((i=0; i<=LAST; i++))
do
  printf "Experiment #%d: " $i
  GPU=0
  while true
  do
    PROCESS=$(nvidia-smi -i $GPU --query-compute-apps=name --format=csv,noheader)
    if [[ "$PROCESS" != *"python"* ]]; then
      break
    fi
    if [ $GPU == 3 ]; then
      sleep 60
    fi
    GPU=$(((GPU + 1) % 4))
  done
  printf "starting on GPU %d\n" $GPU
  python repo/agent/train.py --gpu $GPU --exp $i &>/dev/null &
  sleep 15
done