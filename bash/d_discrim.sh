#!/bin/bash

# best models
python repo/sim/best_models.py

# chunks, simulations, and discriminator input
for part in valid rl
do
  python repo/sim/chunks.py --part $part
  python repo/sim/generate.py --part $part
  python repo/inputs/discrim.py --part $part
done

# run discrim model
for D in {0..7}
do
  printf "Model discrim: "
  GPU=$(bash repo/bash/gpu.sh)
  printf "starting on GPU %d\n" "$GPU"
  python repo/sim/train.py --gpu "$GPU" --name discrim --dropout $D  &>/dev/null &
  sleep 30
done