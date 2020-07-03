# simulator inputs
rclone sync ~/weka/eBay/inputs/featnames dropbox:ebay/data/inputs/featnames
rclone sync ~/weka/eBay/inputs/sizes dropbox:ebay/data/inputs/sizes
rclone copy ~/weka/eBay/inputs/date_feats.pkl dropbox:ebay/data/inputs
rclone sync ~/weka/eBay/outputs/models dropbox:ebay/data/outputs/models --max-depth=1
rclone sync ~/weka/eBay/inputs/valid dropbox:ebay/data/inputs/valid

# grab all of valid chunks
rclone sync ~/weka/eBay/partitions/valid/chunks partitions/valid/chunks

# agent training input files
rclone sync ~/weka/eBay/agent/train dropbox:ebay/data/agent/train
