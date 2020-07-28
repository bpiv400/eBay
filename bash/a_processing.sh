#!/bin/bash

# clean
python repo/processing/a_clean.py

# create features
python repo/processing/b_feats/date_feats.py &
python repo/processing/b_feats/category.py --name leaf &
python repo/processing/b_feats/category.py --name meta &
python repo/processing/b_feats/category.py --name slr
python repo/processing/b_feats/tf.py
python repo/processing/b_feats/w2v.py
python repo/processing/b_feats/w2v.py --slr

# create frames
for part in sim rl valid
do
  for f in lookup lstg thread offer
  do
    python repo/processing/c_frames/$f\.py --part $part &
  done
done