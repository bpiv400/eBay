#!/bin/bash

qsub -N first_listing repo/bash/gen_init_listing.sh $1 toy-1 -js 1 -j y -sync y
count=$(ls data/datasets/$1/listing_chunks/toy -1 | wc -l)
qsub -N other_listings_toy repo/bash/gen_late_listings.sh $1 toy -js 1 -j y -t 2-$count
count=$(ls data/datasets/$1/listing_chunks/train -1 | wc -l)
qsub -N other_listings_train repo/bash/gen_late_listings.sh $1 train -js 1 -j y -t 1-$count
count=$(ls data/datasets/$1/listing_chunks/test -1 | wc -l)
qsub -N other_listings_test repo/bash/gen_late_listings.sh $1 train -js 1 -j y -t 1-$count