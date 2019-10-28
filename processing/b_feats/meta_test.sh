for SGE_TASK_ID in {1..245}
do
  NUM=$((SGE_TASK_ID / 7 + 1))
  FEAT=$((SGE_TASK_ID % 7 + 1))
  echo python repo/processing/b_feats/meta.py --num "$NUM" --feat "$FEAT"
done