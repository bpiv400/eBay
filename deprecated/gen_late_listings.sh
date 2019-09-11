#!/bin/bash
#$ -m e -M 4102158912@vtext.com

scriptPath=repo/rlenv/generate_listings.py
cd ~
source /opt/rh/rh-python36/enable
source ~/envs/bargain/bin/activate
cd eBay
python "$scriptPath" --name $2-"$SGE_TASK_ID" --data $1