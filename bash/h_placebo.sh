#!/bin/bash
# creates the inputs for the placebo discriminator, and trains it with dropout

# chunks, simulations, and discriminator input
for part in valid rl
do
  python repo/inputs/discrim.py --part $part --placebo
done

# run placebo discriminator
for D in {7..14}
do
  printf "Model placebo: "
  GPU=$(bash repo/bash/gpu.sh)
  printf "starting on GPU %d\n" "$GPU"
  python repo/sim/train.py --gpu "$GPU" --name placebo --dropout "$D"  &>/dev/null &
  sleep 30
done