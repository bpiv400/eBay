#!/bin/bash

# no dropout
for NAME in first_arrival next_arrival hist
do
  printf "Model %s: " $NAME
  GPU=$(bash repo/bash/util/gpu.sh)
  printf "starting on GPU %d\n" "$GPU"
  python repo/sim/train.py --gpu "$GPU" --name "$NAME" --dropout 0  &>/dev/null &
  sleep 30
done

# low dropout
for NAME in con1 con2 con3 con4 con5 delay2 delay3 delay4 delay5 msg1 msg2 msg3 policy_slr
do
  for D in {0..1}
  do
    printf "Model %s: " $NAME
    GPU=$(bash repo/bash/util/gpu.sh)
    printf "starting on GPU %d\n" "$GPU"
    python repo/sim/train.py --gpu "$GPU" --name "$NAME" --dropout $D  &>/dev/null &
    sleep 30
  done
done

# moderate dropout
for NAME in con6 delay6 delay7 msg4
do
  for D in {4..8}
  do
    printf "Model %s: " $NAME
    GPU=$(bash repo/bash/util/gpu.sh)
    printf "starting on GPU %d\n" "$GPU"
    python repo/sim/train.py --gpu "$GPU" --name "$NAME" --dropout $D  &>/dev/null &
    sleep 30
  done
done

# high dropout
for NAME in con7 msg5 msg6
do
  for D in {7..14}
  do
    printf "Model %s: " $NAME
    GPU=$(bash repo/bash/util/gpu.sh)
    printf "starting on GPU %d\n" "$GPU"
    python repo/sim/train.py --gpu "$GPU" --name "$NAME" --dropout $D  &>/dev/null &
    sleep 30
  done
done

# best models and training plots
bash repo/bash/util/wait.sh
python repo/sim/best_models.py
python repo/assess/training_curves.py
python repo/plots/training_curves.py

# chunks, discrim generator, and discrim input
for part in valid rl
do
  # python repo/sim/chunks.py --part $part
  python repo/discrim/generate.py --part $part
  python repo/discrim/input.py --part $part
done

# discriminator plots
python repo/assess/unconditional.py
python repo/plots/unconditional.py
python repo/assess/conditional.py
python repo/plots/conditional.py

# run discrim model
for D in {0..7}
do
  printf "Model discrim: "
  GPU=$(bash repo/bash/util/gpu.sh)
  printf "starting on GPU %d\n" "$GPU"
  python repo/sim/train.py --gpu "$GPU" --name discrim --dropout $D  &>/dev/null &
  sleep 30
done

# best discrim model
bash repo/bash/util/wait.sh
python repo/sim/best_models.py --group discrim

# roc plot
python repo/assess/roc.py
python repo/plots/roc.py