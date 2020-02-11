rclone sync dropbox:ebay/data/inputs/featnames inputs/featnames
rclone sync dropbox:ebay/data/inputs/sizes inputs/sizes
rclone copy dropbox:ebay/data/inputs/date_feats.pkl inputs
rclone sync dropbox:ebay/data/outputs/models outputs/models --max-depth=1

# grab specific chunks
rclone copy dropbox:ebay/data/envSimulator/test_rl/chunks/1.gz envSimulator/test_rl/chunks
rclone copy dropbox:ebay/data/envSimulator/test_rl/chunks/1_test.gz envSimulator/test_rl/chunks
