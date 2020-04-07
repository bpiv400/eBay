# simulator inputs
rclone sync ~/weka/eBay/inputs/featnames dropbox:ebay/data/inputs/featnames
rclone sync ~/weka/eBay/inputs/sizes dropbox:ebay/data/inputs/sizes
rclone copy ~/weka/eBay/inputs/date_feats.pkl dropbox:ebay/data/inputs
rclone sync ~/weka/eBay/outputs/models dropbox:ebay/data/outputs/models --max-depth=1

# grab all of 1 chunk and the test data from test_rl
rclone copy ~/weka/eBay/envSimulator/test_rl/chunks/1.gz dropbox:ebay/data/envSimulator/test_rl/chunks
rclone copy ~/weka/eBay/envSimulator/test_rl/chunks/1_test.gz dropbox:ebay/data/envSimulator/test_rl/chunks


# agent training input files
rclone copy ~/weka/eBay/agent/train/seller.hdf5 dropbox:ebay/data/agent/train
rclone sync ~/weka/eBay/agent/eval dropbox:ebay/data/agent/eval
