#!/bin/bash

# clean
python repo/processing/a_clean.py

# create features
python repo/processing/b_feats/date_feats.py
python repo/processing/b_feats/w2v.py
python repo/processing/b_feats/w2v.py --slr
python repo/processing/b_feats/category.py --name leaf &
python repo/processing/b_feats/category.py --name meta
python repo/processing/b_feats/category.py --name slr
python repo/processing/b_feats/tf.py

# create frames
for part in sim discrim rl_byr rl_slr valid
do
  for f in lookup lstg thread offer
  do
    python repo/processing/c_frames/$f\.py --part $part &
  done
done