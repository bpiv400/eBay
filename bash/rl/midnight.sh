#!/bin/bash

SECONDS=$(((24 * 3600) - $(date -d "1970-01-01 UTC $(date +%T)" +%s)))
printf "Waiting %d seconds until midnight" $SECONDS
sleep $SECONDS