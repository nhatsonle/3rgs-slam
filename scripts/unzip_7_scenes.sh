dest="datasets/7-scenes"
for scene in chess fire office; do
  unzip -o "$dest/${scene}.zip" -d "$dest"
  for seqzip in "$dest/$scene"/seq-*.zip; do
    [ -f "$seqzip" ] && unzip -o "$seqzip" -d "$dest/$scene"
  done
done