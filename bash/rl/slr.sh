#!/bin/bash

while getopts ":f:" opt; do
  case $opt in
    f) FEAT_TYPE="$OPTARG"
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
  if [ "$FEAT_TYPE" == "" ]; then
    python repo/agent/train.py --gpu "$GPU" --exp $i --log &>/dev/null &
  else
    python repo/agent/train.py --gpu "$GPU" --exp $i --log --feat_type "$FEAT_TYPE" &>/dev/null &
  fi
  sleep 15
done