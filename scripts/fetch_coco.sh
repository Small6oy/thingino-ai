#!/usr/bin/env bash
# Fetch the official COCO 2017 dataset (train/val/test images + annotations) into a local store.
# Usage: scripts/fetch_coco.sh [DEST] [--extract]
#   DEST defaults to $COCO_ROOT or /mnt/models/datasets/coco2017.
#   Without --extract it downloads, integrity-checks and records checksums, leaving the zips ready.
set -euo pipefail

dest=${COCO_ROOT:-/mnt/models/datasets/coco2017}
extract=0
for a in "$@"; do
  case "$a" in
    --extract) extract=1 ;;
    -*) echo "unknown option: $a" >&2; exit 2 ;;
    *) dest=$a ;;
  esac
done

dl="$dest/downloads"
mkdir -p "$dl"

base=http://images.cocodataset.org
files=(zips/train2017.zip zips/val2017.zip zips/test2017.zip annotations/annotations_trainval2017.zip)

if command -v aria2c >/dev/null 2>&1; then
  for f in "${files[@]}"; do
    aria2c -c -x 8 -s 8 --summary-interval=60 -d "$dl" "$base/$f"
  done
else
  for f in "${files[@]}"; do
    z="$dl/$(basename "$f")"
    [ -f "$z" ] || curl -fL --retry 5 -C - -o "$z" "$base/$f"
  done
fi

for f in "${files[@]}"; do
  z="$dl/$(basename "$f")"
  echo "checking $z"
  unzip -tq "$z" >/dev/null
done

python3 - "$dl" <<'PY'
import json, os, sys, zipfile

dl = sys.argv[1]
expected = {'train2017.zip': 118287, 'val2017.zip': 5000, 'test2017.zip': 40670}
for z, n in expected.items():
    with zipfile.ZipFile(os.path.join(dl, z)) as zf:
        images = [x for x in zf.namelist() if x.lower().endswith('.jpg')]
    if len(images) != n:
        raise SystemExit('%s: %d images, expected %d' % (z, len(images), n))
    print('%s: %d images OK' % (z, len(images)))

ann = os.path.join(dl, 'annotations_trainval2017.zip')
with zipfile.ZipFile(ann) as zf:
    for name, n in (('annotations/instances_train2017.json', 118287),
                    ('annotations/instances_val2017.json', 5000)):
        d = json.loads(zf.read(name))
        if len(d['images']) != n:
            raise SystemExit('%s: %d images, expected %d' % (name, len(d['images']), n))
        print('%s: %d images OK' % (name, len(d['images'])))
PY

( cd "$dl" && sha256sum ./*.zip > SHA256SUMS )
echo "checksums: $dl/SHA256SUMS"

if [ "$extract" -eq 1 ]; then
  for f in "${files[@]}"; do
    z="$dl/$(basename "$f")"
    if command -v ionice >/dev/null 2>&1; then
      nice -n 19 ionice -c 3 unzip -qo "$z" -d "$dest"
    else
      nice -n 19 unzip -qo "$z" -d "$dest"
    fi
  done
  echo "extracted to $dest"
else
  echo "zips ready. Re-run with --extract (ideally after the current training) to unpack."
fi
