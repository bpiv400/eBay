#!/bin/bash
cd ~/eBay/data/
count=$(ls $1 -1q $1* | wc -l)
count="$(($count-1))"
for i in {0..$count}; do
	j="$(($i+1))"
	mv $1/$1-$i.csv $1/$1-$j.csv
done
