# simulator inputs
rclone sync dropbox:ebay/data/inputs/featnames data/inputs/featnames
rclone sync dropbox:ebay/data/inputs/sizes data/inputs/sizes
rclone copy dropbox:ebay/data/inputs/date_feats.pkl data/inputs
rclone sync dropbox:ebay/data/outputs/models data/outputs/models --max-depth=1
rclone sync dropbox:ebay/data/inputs/valid data/inputs/vallid

# grab all of valid chunks
rclone sync dropbox:ebay/data/partitions/valid/chunks data/partitions/valid/chunks
rclone sync dropbox:ebay/data/partitions/valid data/partitions/valid --max-depth=1
# agent files
rclone sync dropbox:ebay/data/agent/train data/agent/train
