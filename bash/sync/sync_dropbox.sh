# simulator inputs
rclone sync ~/weka/eBay/inputs/featnames dropbox:ebay/data/inputs/featnames
rclone sync ~/weka/eBay/inputs/sizes dropbox:ebay/data/inputs/sizes
rclone copy ~/weka/eBay/inputs/date_feats.pkl dropbox:ebay/data/inputs
rclone sync ~/weka/eBay/outputs/models dropbox:ebay/data/outputs/models --max-depth=1

# testing framework inputs
rclone sync ~/weka/eBay/inputs/valid dropbox:ebay/data/inputs/valid
rclone sync ~/weka/eBay/index/valid dropbox:ebay/data/index/valid
rclone copy ~/weka/eBay/feats/offers.pkl dropbox:ebay/data/feats

# grab all of valid chunks
rclone sync ~/weka/eBay/partitions/agent/valid/chunks dropbox:ebay/data/partitions/agent/valid/chunks
# TODO: ADD BACK
# rclone sync ~/weka/eBay/partitions/models/valid/chunks dropbox:ebay/partitions/models/valid/chunks
# x_offer, x_thread, x_lstg, clock, lookup
rclone sync ~/weka/eBay/partitions/agent/valid dropbox:ebay/data/partitions/agent/valid --max-depth=1

# agent training input files
rclone sync ~/weka/eBay/partitions/rl/agent dropbox:ebay/data/partitions/rl/agent
