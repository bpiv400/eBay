#!/bin/bash

# first arrival model in all partitions
for part in sim rl_byr rl_slr valid; do
  python repo/inputs/first_arrival.py --part $part &
done

# for other inputs, skip rl partitions
for part in sim valid; do
  for f in next_arrival hist policy_slr policy_byr; do
    python repo/inputs/$f\.py --part $part &
  done
  for turn in {1..7}; do
    python repo/inputs/offer.py --part $part --turn $turn --outcome con &
    if [ $turn -gt 1 ]; then
      python repo/inputs/offer.py --part $part --turn $turn --outcome delay &
    fi
    if [ $turn -lt 7 ]; then
      python repo/inputs/offer.py --part $part --turn $turn --outcome msg &
    fi
  done
done