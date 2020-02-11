rclone sync ~/weka/eBay/inputs/featnames dropbox:ebay/data/inputs/featnames
rclone sync ~/weka/eBay/inputs/sizes dropbox:ebay/data/inputs/sizes
rclone copy ~/weka/eBay/inputs/date_feats.pkl dropbox:ebay/data/inputs
rclone sync ~/weka/eBay/outputs/models dropbox:ebay/data/outputs/models --max-depth=1

# grab specific chunks from train rl
rclone copy ~/weka/eBay/envSimulator/test_rl/chunks/1.gz dropbox:ebay/data/envSimulator/test_rl/chunks
rclone copy ~/weka/eBay/envSimulator/test_rl/chunks/1_test.gz dropbox:ebay/data/envSimulator/test_rl/chunks
