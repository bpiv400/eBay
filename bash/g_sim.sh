#!/bin/bash

# chunks, simulations, and discriminator input
for n in {1..1024}
do
  # seller agent
  for DELTA in {0.0,0.75}
  do
    python repo/agent/eval/sims.py --num "$n" --delta "$DELTA" &
  done

  # seller agent heuristic
  for DELTA in {0.0,0.75}
  do
    python repo/agent/eval/sims.py --num "$n" --delta "$DELTA" --heuristic &
  done

  # buyer agent
  for DELTA in {0.9,1.0,1.5,2.0,3.0}
  do
    python repo/agent/eval/sims.py --num "$n" --byr --delta "$DELTA" &
  done

  # buyer heuristics
  for INDEX in {0..214}
  do
    python repo/agent/eval/sims.py --num "$n" --byr --heuristic --index "$INDEX" &
  done

  # for placebo discriminator
  python repo/agent/eval/sims.py --part rl --num "$n" --byr --heuristic --index 33 &
done

