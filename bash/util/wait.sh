#!/bin/bash

printf "Waiting for compute processes to finish...\n"
GPU=0
while true
do
  PID=$(nvidia-smi -i $GPU | grep '      C   ' | awk '{ print $3 }')
  if [[ "$PID" == "" ]]; then
    GPU=$((GPU + 1))
  else
    GPU=0
    sleep 60
  fi

  if [[ $GPU == 4 ]]; then
    break
  fi
done