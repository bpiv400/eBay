#!/bin/bash
org=$1
new=$2
cd ~/eBay/data/exps
cp ~/eBay/data/exps/$org ~/eBay/data/exps/$new
cd ~/eBay/interface/exps
mkdir $new
