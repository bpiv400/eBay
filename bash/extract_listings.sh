dir=$1
type=$2
cd data/datasets/$(dir)/listings/$(type)
dir -1 | sed 's/.pkl$//' > ./../$(type)_listings.txt
cd ~/eBay