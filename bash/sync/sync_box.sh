# simulator inputs
rclone sync dropbox:ebay/data/inputs/featnames inputs/featnames
rclone sync dropbox:ebay/data/inputs/sizes inputs/sizes
rclone copy dropbox:ebay/data/inputs/date_feats.pkl inputs
rclone sync dropbox:ebay/data/outputs/models outputs/models --max-depth=1

# grab all of test_rl
for partition in test_rl train_rl
do
  rclone copy dropbox:ebay/data/partitions/${partition}/chunks/1.gz partitions/${partition}/chunks
done
rclone copy dropbox:ebay/data/partitions/test_rl/chunks/1_test.gz partitions/test_rl/chunks

# agent files
rclone sync dropbox:ebay/data/agent/train agent/train
