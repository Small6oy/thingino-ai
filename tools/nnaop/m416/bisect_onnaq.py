#!/usr/bin/env python3
# Copyright (C) 2026 Thingino Project
# SPDX-License-Identifier: MIT
"""First tensor where onnaq differs from Ingenic's PC executor (per-layer bisect).

  python3 mkgraphq.py MODEL.onnx --bin MODEL.bin --names DUMPDIR/names.txt > G.qgraph
  ONNAQ_DUMP=DUMPDIR onnaq -e cpu MODEL.bin G.qgraph FRAME.rgb out
  QGRAPH=G.qgraph bisect_onnaq.py DUMPDIR DUMPDIR/names.txt MODEL.onnx MODEL_quant.mgk FRAME.rgb [max]

Walks onnaq's tensors in graph order, dumps the same names from the executor (mgkdump in the
magik Docker image; MGKDUMP, MAGIK_IMAGE as in mgkcmp.sh) and prints ok / the first differences.
A conv's tensor is compared under its ReLU's name (the executor's).
"""
import glob
import os
import struct
import subprocess
import sys

import numpy as np
import onnx

d, names_f, onnx_p, mgk, frame = sys.argv[1:6]
mx = int(sys.argv[6]) if len(sys.argv) > 6 else 400
name_of = {}
for line in open(names_f):
    i, n = line.split(" ", 1)
    name_of[int(i)] = n.strip()
m = onnx.load(onnx_p)
relu_after = {n.input[0]: n.output[0] for n in m.graph.node if n.op_type == "Relu"}
have = {int(os.path.basename(f)[1:-3]) for f in glob.glob(d + "/t*.u8")}
order = []
for line in open(os.environ["QGRAPH"]):          # op order = the graph's line order
    f = line.split()
    if len(f) > 1 and f[0] not in ("onnaq-graph", "tensors", "input", "output", "anchors"):
        t = int(f[1])
        if t in have and t not in order:
            order.append(t)
order = order[:mx]
want = {}
for tid in order:
    n = name_of.get(tid)
    if n is None:
        continue
    want[tid] = relu_after.get(n, n)
ex = d + "/ex"
os.makedirs(ex, exist_ok=True)
names = sorted(set(want.values()))
home = os.path.expanduser("~")
mgkdump = os.environ.get("MGKDUMP", home + "/nnaopen/m416/pc/mgkdump")
image = os.environ.get("MAGIK_IMAGE", "thingino-ai-magik:2.0.4")
def dump(batch):
    """mgkdump aborts the whole run on a name the converted model lacks: split failed batches"""
    vols = []
    for p in sorted({os.path.dirname(os.path.abspath(x)) for x in (mgk, frame, mgkdump)} | {os.path.abspath(ex)}):
        vols += ["-v", "%s:%s" % (p, p)]                 # the same paths inside the container
    subprocess.run(["docker", "run", "--rm"] + vols + ["-e", "LD_LIBRARY_PATH=/opt/magik/pci/lib",
                    image, os.path.abspath(mgkdump), os.path.abspath(mgk), "320", "416", os.path.abspath(frame),
                    os.path.abspath(ex)] + batch, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if len(batch) > 1 and not all(os.path.exists("%s/%s.f32" % (ex, n)) for n in batch):
        dump(batch[:len(batch) // 2])
        dump(batch[len(batch) // 2:])


for k in range(0, len(names), 60):
    dump(names[k:k + 60])
bad = 0
for tid in order:
    n = want.get(tid)
    if n is None:
        continue
    b = open("%s/t%d.u8" % (d, tid), "rb").read()
    h, w, c = struct.unpack_from("<3i", b, 0)
    ours = np.frombuffer(b, np.uint8, h * w * c, 12).astype(np.float32)
    ef = "%s/%s.f32" % (ex, n)
    if not os.path.exists(ef):
        print("t%-4d %-16s (no executor tensor)" % (tid, n))
        continue
    theirs = np.fromfile(ef, np.float32)
    if theirs.size != ours.size:
        print("t%-4d %-16s size %d vs %d" % (tid, n, ours.size, theirs.size))
        continue
    diff = int((ours != theirs).sum())
    if diff:
        print("t%-4d %-16s %dx%dx%d: %d of %d differ (max |diff| %g)  <-- first difference" % (tid, n, h, w, c, diff, ours.size, np.abs(ours - theirs).max()))
        bad += 1
        if bad >= 3:
            break
    else:
        print("t%-4d %-16s ok" % (tid, n))
