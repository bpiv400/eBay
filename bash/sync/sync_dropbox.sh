# simulator inputs
rclone sync ~/weka/eBay/inputs/featnames dropbox:ebay/data/inputs/featnames
rclone sync ~/weka/eBay/inputs/sizes dropbox:ebay/data/inputs/sizes
rclone copy ~/weka/eBay/inputs/date_feats.pkl dropbox:ebay/data/inputs
rclone sync ~/weka/eBay/outputs/models dropbox:ebay/data/outputs/models --max-depth=1

# grab 1 chunk from test_rl and train_rl
for partition in test_rl train_rl
do
  rclone copy ~/weka/eBay/partitions/${partition}/chunks/1.gz dropbox:ebay/data/partitions/${partition}/chunks
done
rclone copy ~/weka/eBay/partitions/test_rl/chunks/1_test.gz dropbox:ebay/data/partitions/test_rl/chunks


# agent training input files
rclone sync ~/weka/eBay/agent/train dropbox:ebay/data/agent/train
