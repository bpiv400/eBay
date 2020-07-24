#!/bin/bash

# best models
bash repo/bash/wait.sh
python repo/sim/best_models.py

# chunks and discrim generator
for part in valid rl
do
  GPU=$(bash repo/bash/gpu.sh)
  printf "Running discrim generator for %s" "$part"
  python repo/sim/post.py --part $part --gpu "$GPU" &>/dev/null &
done

# discriminator input
bash repo/bash/wait.sh
python repo/inputs/discrim.py --part valid &
python repo/inputs/discrim.py --part rl

# run discrim model
for D in {0..1}
do
  printf "Model discrim: "
  GPU=$(bash repo/bash/gpu.sh)
  printf "starting on GPU %d\n" "$GPU"
  python repo/sim/train.py --gpu "$GPU" --name discrim --dropout $D  &>/dev/null &
  sleep 30
done

bash repo/bash/wait.sh
python repo/sim/best_models.py --discrim