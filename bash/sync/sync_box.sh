# simulator inputs
rclone sync dropbox:ebay/data/inputs/featnames inputs/featnames
rclone sync dropbox:ebay/data/inputs/sizes inputs/sizes
rclone copy dropbox:ebay/data/inputs/date_feats.pkl inputs
rclone sync dropbox:ebay/data/outputs/models outputs/models --max-depth=1
rclone sync dropbox:ebay/data/inputs/valid inputs/vallid

# grab all of valid chunks
rclone sync dropbox:ebay/data/partitions/valid/chunks partitions/valid/chunks

# agent files
rclone sync dropbox:ebay/data/agent/train agent/train
