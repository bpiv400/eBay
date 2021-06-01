#!/bin/bash

# create features
python repo/processing/a_feats/other.py
python repo/processing/a_feats/w2v.py
python repo/processing/a_feats/w2v.py --slr
python repo/processing/a_feats/category.py --name leaf &
python repo/processing/a_feats/category.py --name meta
python repo/processing/a_feats/category.py --name slr
python repo/processing/a_feats/tf.py

# create frames
for part in sim rl valid
do
  for f in lookup lstg thread offer
  do
    python repo/processing/b_frames/$f\.py --part $part &
  done
done