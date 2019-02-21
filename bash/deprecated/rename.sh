#!/bin/bash
cd ~/eBay/data/$1
count=$(ls -1q $1* | wc -l)
count="$(($count-1))"
for (( i=$count; i>=0; i-- )) do
	j="$(($i+1))"
	mv $1-$i.csv $1-$j.csv
done
