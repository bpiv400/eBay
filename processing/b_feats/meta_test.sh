for SGE_TASK_ID in {1..245}
do
  NUM=$((SGE_TASK_ID / 7))
  FEAT=$((7 - SGE_TASK_ID % 7))
  echo python repo/processing/b_feats/meta.py --num "$NUM" --feat "$FEAT"
done