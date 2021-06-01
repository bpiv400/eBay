#!/bin/bash

# chunks, simulations, and discriminator input
for part in valid rl
do
  python repo/sim/collate.py --part $part
  python repo/sim/collate.py --part $part --values
  python repo/inputs/discrim.py --part $part
done

# run discriminator
for D in {0..7}
do
  printf "Model discrim: "
  GPU=$(bash repo/bash/gpu.sh)
  printf "starting on GPU %d\n" "$GPU"
  python repo/sim/train.py --gpu "$GPU" --name discrim --dropout "$D"  &>/dev/null &
  sleep 30
done