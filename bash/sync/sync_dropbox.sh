rclone sync inputs/featnames dropbox:ebay/data/inputs/featnames
rclone sync inputs/sizes dropbox:ebay/data/inputs/sizes
rclone copy inputs/params.pkl dropbox:ebay/data/inputs
rclone copy inputs/date_feats.pkl dropbox:ebay/data/inputs
rclone copy envSimulator/train_rl/chunks/1.gz dropbox:ebay/data/envSimulator/train_rl/chunks
rclone sync outputs/models outputs/models --max-depth=1