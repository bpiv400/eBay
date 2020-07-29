#!/bin/bash

# default values
DELTA=0.995
BETA=1.0
OTHER_ARGUMENTS=()

# Loop through arguments and process them
for arg in "$@"
do
    case $arg in
        -b|--beta)
        BETA="$2"
        shift # Remove argument name from processing
        shift # Remove argument value from processing
        ;;
        -d|--delta)
        DELTA="$2"
        shift # Remove argument name from processing
        shift # Remove argument value from processing
        ;;
        *)
        OTHER_ARGUMENTS+=("$1")
        shift # Remove generic argument from processing
        ;;
    esac
done

# with entropy bonus
./repo/bash/util/wait.sh
GPU=$(bash repo/bash/util/gpu.sh)
printf "entropy: starting on cuda:%d\n" "$GPU"
python repo/agent/train.py --gpu "$GPU" --log --all --delta "$DELTA" --beta "$BETA" &>/dev/null &
sleep 30

# with kl penalty
./repo/bash/util/wait.sh
for kl in 0 1.0 0.1 0.01 0.001 0.0001
do
  GPU=$(bash repo/bash/util/gpu.sh)
  printf "kl %1.4g: starting on cuda:%d\n" "$kl" "$GPU"
  python repo/agent/train.py --gpu "$GPU" --log --delta "$DELTA" --beta "$BETA" --kl $kl &>/dev/null &
  sleep 30
done