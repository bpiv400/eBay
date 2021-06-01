#!/bin/bash

GPU=0
while true
do
  MB=$(nvidia-smi -i $GPU | grep '      C   ' | awk '{ print $8 }' | uniq)
  if [[ "$MB" == "0MiB" || "$MB" == "" ]]; then
    break
  fi
  if [ $GPU == 3 ]; then
    sleep 60
  fi
  GPU=$(((GPU + 1) % 4))
done
echo "$GPU"