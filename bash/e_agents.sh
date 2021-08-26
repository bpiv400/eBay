#!/bin/bash

# seller models
for DELTA in {0.0,0.75}
do
  printf "Seller %s: " "$DELTA"
  GPU=$(bash repo/bash/gpu.sh)
  printf "starting on GPU %d\n" "$GPU"
  python repo/agent/train.py --gpu "$GPU" --log --delta "$DELTA"  &>/dev/null &
  sleep 30
done

# buyer models
for DELTA in {0.9,1.0,1.5,2.0,3.0}
do
  printf "Buyer %s: " "$DELTA"
  GPU=$(bash repo/bash/gpu.sh)
  printf "starting on GPU %d\n" "$GPU"
  python repo/agent/train.py --gpu "$GPU" --log --byr --delta "$DELTA"  &>/dev/null &
  sleep 30
done

# buyer models
for COST in {1,2,3,4}
do
  printf "Buyer turn cost %s: " "$COST"
  GPU=$(bash repo/bash/gpu.sh)
  printf "starting on GPU %d\n" "$GPU"
  python repo/agent/train.py --gpu "$GPU" --log --byr --delta 1 --turn_cost "$COST"  &>/dev/null &
  sleep 30
done