# simulator inputs
rclone sync dropbox:ebay/data/inputs/featnames data/inputs/featnames
rclone sync dropbox:ebay/data/inputs/sizes data/inputs/sizes
rclone copy dropbox:ebay/data/feats/date_feats.pkl data/feats
rclone sync dropbox:ebay/data/outputs/models data/outputs/models --max-depth=1

# testing framework inputs
rclone sync dropbox:ebay/data/inputs/valid data/inputs/valid
rclone sync dropbox:ebay/data/index/valid data/index/valid
rclone copy dropbox:ebay/data/feats/offers.pkl data/feats

# grab all of valid chunks for testing framework
rclone sync dropbox:ebay/data/partitions/agent/valid/chunks data/partitions/agent/valid/chunks
# TODO: ADD BLACK
# rclone sync dropbox:ebay/data/partitions/models/valid/chunks data/partitions/model/valid/chunks
# x_offer, x_thread, x_lstg, clock, lookup
rclone sync dropbox:ebay/data/partitions/agent/valid data/partitions/agent/valid --max-depth=1
rclone sync dropbox:ebay/data/partitions/models/valid data/partitions/models/valid --max-depth=1

# agents files
rclone sync dropbox:ebay/data/partitions/agent/rl/chunks data/partitions/agent/rl/chunks

