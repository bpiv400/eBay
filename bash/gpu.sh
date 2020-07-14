#!/bin/bash

GPU=0
while true
do
  PID=$(nvidia-smi -i $GPU | grep '      C   ' | awk '{ print $3 }')
  if [[ "$PID" == "" ]]; then
    break
  fi
  if [ $GPU == 3 ]; then
    sleep 60
  fi
  GPU=$(((GPU + 1) % 4))
done
echo "$GPU"