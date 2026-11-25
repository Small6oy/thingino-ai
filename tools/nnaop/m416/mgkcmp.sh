#!/bin/sh
# Copyright (C) 2026 Thingino Project
# SPDX-License-Identifier: MIT
# mgkcmp.sh MODEL.onnx MODEL.bin MODEL_quant.mgk FRAME.rgb [GRAPH|-]
# onnaq -e cpu against Ingenic's PC executor (mgkdump.cc, run in the magik Docker image), head by
# head. GRAPH: a .qgraph, or - (default) for the graph from the .bin's own op list. The ONNX only
# names the heads (the executor calls them <onnx output>_output). Frames are 416x320 (H W from
# the .bin). Environment:
#   ONNAQ        onnaq binary              [../../onna/onnaq next to this script]
#   MGKDUMP      mgkdump built from mgkdump.cc against the image's libmagikExecutor  [$HOME/nnaopen/m416/pc/mgkdump]
#   MAGIK_IMAGE  Docker image with /opt/magik/pci/lib  [thingino-ai-magik:2.0.4]
#   PY           python with onnx + numpy  [python3]
set -eu
[ $# -ge 4 ] || { sed -n '4,13p' "$0"; exit 2; }
onnx=$1 bin=$2 mgk=$3 frame=$4 graph=${5:--}
here=$(cd "$(dirname "$0")" && pwd)
ONNAQ=${ONNAQ:-$here/../../onna/onnaq}
MGKDUMP=${MGKDUMP:-$HOME/nnaopen/m416/pc/mgkdump}
MAGIK_IMAGE=${MAGIK_IMAGE:-thingino-ai-magik:2.0.4}
PY=${PY:-python3}
w=$(mktemp -d)
trap 'rm -rf "$w"' EXIT
outs=$("$PY" -c "import onnx,sys; print(' '.join(o.name + '_output' for o in onnx.load(sys.argv[1]).graph.output))" "$onnx")
for f in "$bin" "$mgk" "$frame" "$MGKDUMP"; do [ -e "$f" ] || { echo "missing $f" >&2; exit 1; }; done
abs() { (cd "$(dirname "$1")" && echo "$(pwd)/$(basename "$1")"); }
docker run --rm -v "$(dirname "$(abs "$mgk")"):/m:ro" -v "$(dirname "$(abs "$frame")"):/f:ro" \
	-v "$(dirname "$(abs "$MGKDUMP")"):/x:ro" -v "$w:/w" -e LD_LIBRARY_PATH=/opt/magik/pci/lib "$MAGIK_IMAGE" \
	/x/"$(basename "$MGKDUMP")" /m/"$(basename "$mgk")" 320 416 /f/"$(basename "$frame")" /w $outs | grep -v INFO || true
"$ONNAQ" -e cpu "$bin" "$graph" "$frame" "$w/oq" 2>&1 | grep -E "onnaq: [^0-9]|runs?:" || true
"$PY" - "$w" $outs <<'PY'
import sys, glob, numpy as np
w, outs = sys.argv[1], sys.argv[2:]
g = {o: np.fromfile("%s/%s.f32" % (w, o), np.float32) for o in outs}
bad = 0
for f in sorted(glob.glob(w + "/oq.*.f32")):
    o = np.fromfile(f, np.float32)
    for k, v in g.items():
        if v.size == o.size:
            same = np.array_equal(o, v)
            bad += not same
            print("head %s vs %s: %s (%d of %d differ)" % (f.rsplit(".", 2)[-2], k, "identical" if same else "DIFFERENT", int((o != v).sum()), v.size))
sys.exit(1 if bad else 0)
PY
