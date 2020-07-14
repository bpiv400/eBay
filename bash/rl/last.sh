#!/bin/bash

LAST=$(tail -n1 "data/outputs/agent/exps.csv" | cut -d, -f1)
echo "$LAST"