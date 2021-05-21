#!/bin/bash

# best models
python repo/sim/best_models.py

# chunks, simulations, and discriminator input
for part in valid rl
do
  python repo/sim/synthetic.py --part $part
  python repo/sim/chunks.py --part $part
  python repo/sim/sims.py --part $part
  python repo/inputs/discrim.py --part $part
  python repo/inputs/discrim.py --part $part --placebo
done

# run discriminator model
for D in {0..7}
do
  printf "Model discrim: "
  GPU=$(bash repo/bash/gpu.sh)
  printf "starting on GPU %d\n" "$GPU"
  python repo/sim/train.py --gpu "$GPU" --name discrim --dropout "$D"  &>/dev/null &
  sleep 30
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