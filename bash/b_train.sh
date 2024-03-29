#!/bin/bash
# trains simulator models, varying dropout rates

# no dropout
for NAME in first_arrival next_arrival hist
do
  printf "Model %s: " $NAME
  GPU=$(bash repo/bash/gpu.sh)
  printf "starting on GPU %d\n" "$GPU"
  python repo/sim/train.py --gpu "$GPU" --name "$NAME" --dropout 0  &>/dev/null &
  sleep 30
done

# low dropout
for NAME in con1 con2 con3 con4 con5 delay2 delay3 delay4 delay5 msg1 msg2 msg3
do
  for D in {0..1}
  do
    printf "Model %s: " $NAME
    GPU=$(bash repo/bash/gpu.sh)
    printf "starting on GPU %d\n" "$GPU"
    python repo/sim/train.py --gpu "$GPU" --name "$NAME" --dropout "$D"  &>/dev/null &
    sleep 30
  done
done

# moderate dropout
for NAME in con6 delay6 delay7 msg4
do
  for D in {4..8}
  do
    printf "Model %s: " $NAME
    GPU=$(bash repo/bash/gpu.sh)
    printf "starting on GPU %d\n" "$GPU"
    python repo/sim/train.py --gpu "$GPU" --name "$NAME" --dropout "$D"  &>/dev/null &
    sleep 30
  done
done

# high dropout
for NAME in con7 msg5 msg6
do
  for D in {7..12}
  do
    printf "Model %s: " $NAME
    GPU=$(bash repo/bash/gpu.sh)
    printf "starting on GPU %d\n" "$GPU"
    python repo/sim/train.py --gpu "$GPU" --name "$NAME" --dropout "$D"  &>/dev/null &
    sleep 30
  done
done