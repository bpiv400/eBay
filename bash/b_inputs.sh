#!/bin/bash

# first arrival model in all partitions
for part in sim rl valid; do
  python repo/inputs/first_arrival.py --part $part &
done

# for other model inputs, skip rl partitions
for part in sim valid; do
  for f in next_arrival hist; do
    python repo/inputs/$f\.py --part $part &
  done
  for turn in {1..7}; do
    python repo/inputs/concession.py --part $part --turn $turn &
    if [ $turn -gt 1 ]; then
      python repo/inputs/delay.py --part $part --turn $turn &
    fi
    if [ $turn -lt 7 ]; then
      python repo/inputs/msg.py --part $part --turn $turn &
    fi
  done
done

# for testing agent environments: valid partition for agent inputs
python repo/inputs/agents.py &
python repo/inputs/agents.py --byr &