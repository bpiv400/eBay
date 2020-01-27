rclone sync dropbox:ebay/data/inputs/featnames inputs/featnames
rclone sync dropbox:ebay/data/inputs/sizes inputs/sizes
rclone copy dropbox:ebay/data/inputs/date_feats.pkl inputs
rclone copy dropbox:ebay/data/envSimulator/train_rl/chunks/1.gz envSimulator/train_rl/chunks
rclone sync dropbox:ebay/data/outputs/models outputs/models --max-depth=1
