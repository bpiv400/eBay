#!/bin/bash
# collates the agent evaluation output

# seller agent
for DELTA in {0.0,0.75}
do
  python repo/agent/eval/collate.py --delta "$DELTA" &
done

# seller agent heuristic
for DELTA in {0.0,0.75}
do
  python repo/agent/eval/collate.py --delta "$DELTA" --heuristic &
done

# buyer agent
for DELTA in {0.9,1.0,1.5,2.0,3.0}
do
  python repo/agent/eval/collate.py --byr --delta "$DELTA" &
done

# buyer heuristics
for INDEX in {0..214}
do
  python repo/agent/eval/collate.py --byr --heuristic --index "$INDEX" &
done

# for placebo discriminator
python repo/agent/eval/collate.py --part rl --byr --heuristic --index 33 &