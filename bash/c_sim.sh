#!/bin/bash

DIR=$(<data_folder.txt)

# best models
python repo/sim/best_models.py

# chunks, simulations, and discriminator input
for part in valid rl
do
  python repo/sim/synthetic.py --part $part

  for n in {1..1024}
  do
    python repo/sim/chunks.py --part $part --num "$n" &
  done

  for n in {1..1024}
  do
    CHUNK="${DIR}partitions/${part}/chunks/${n}.pkl"
    if test -f "$CHUNK"; then
      if [ "$n" == 1 ]; then  # run testing for one chunk
        python repo/testing/test.py --num "$n"
        python repo/testing/test.py --num "$n" --byr
        python repo/testing/test.py --num "$n" --slr
      fi
      python repo/sim/sims.py --part $part --num "$n" &
      python repo/sim/sims.py --part $part --values --num "$n" &
    fi
  done
done