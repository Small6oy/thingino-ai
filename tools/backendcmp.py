#!/usr/bin/env python3
# Copyright (C) 2026 Thingino Project
# SPDX-License-Identifier: MIT
"""Score one T41 model on several backends with the same frames: does the runtime or the converter
cost detections?

  backendcmp.py frames --data DATASET [--n 300] OUT          pick validation frames with people
  backendcmp.py onnaq  MODEL.bin OUT/onna [--qgraph G]       run onnaq -e cpu on every frame
  backendcmp.py score  --data DATASET OUT SRC [SRC ...]      AP50 + person threshold sweep

frames: every 7th validation image that has a person, preprocessed as training does it (centre
16:9 crop stretched to 416x320, RGB), written as OUT/fNNNN.rgb plus OUT/frames.u8 (all frames, one
file, for mgkrun) and OUT/frames.json (the dataset indices, so score finds the labels again).

SRC is either a folder of onnaq outputs (fNNNN.0.f32, fNNNN.1.f32, as the onnaq subcommand writes)
or an mgkrun output prefix (PREFIX.<output>.f32, one file per head, N frames each):

  mgkrun model_quant.mgk 320 416 OUT/frames.u8 OUT/m204/out      Ingenic's PC executor
                                                                  (train/t41/mgkrun.cc)

mgkrun.cc builds against any Magik PC executor with the same API, e.g. the venus 2.0.4 one in
train/t41/Dockerfile or a Magik 3.x kit's Models/samples/pc-inference (include/ + lib/):

  g++ -O2 -std=c++11 -mavx -I$PCI/include train/t41/mgkrun.cc -o mgkrun -L$PCI/lib \\
      -lmagikExecutor -lprotobuf -lprotobuf-lite -Wl,-rpath,$PCI/lib
  LD_LIBRARY_PATH=$PCI/lib ./mgkrun ...

score decodes every source with train/qat/train.py decode() (classes from the head width, strides
from the grid, the default anchors unless --anchors) and prints person AP50, the other classes'
AP50, and for frames with one or two people of at least --min-px: people found at IoU 0.5 and
false person boxes per frame, at thresholds 0.10 .. 0.40. --json writes the same.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'train', 'qat'))
H, W = 320, 416
THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)


def dataset(root):
    from train import DetSet
    return DetSet(root, 'val', False)


def cmd_frames(a):
    ds = dataset(a.data)
    os.makedirs(a.out, exist_ok=True)
    idx = []
    for i in range(0, len(ds), 7):
        if len(idx) >= a.n:
            break
        if (ds[i][1][:, 1] == 0).any():
            idx.append(i)
    with open(os.path.join(a.out, 'frames.u8'), 'wb') as all_:
        for k, i in enumerate(idx):
            f = ds[i][0].permute(1, 2, 0).numpy().astype(np.uint8)
            assert f.shape == (H, W, 3), f.shape
            f.tofile(os.path.join(a.out, 'f%04d.rgb' % k))
            all_.write(f.tobytes())
    json.dump(dict(dataset=os.path.abspath(a.data), split='val', indices=idx),
              open(os.path.join(a.out, 'frames.json'), 'w'))
    print('%d frames -> %s' % (len(idx), a.out))


def cmd_onnaq(a):
    onnaq = a.onnaq or os.path.join(HERE, 'onna', 'onnaq')
    fdir = os.path.dirname(os.path.abspath(a.out))
    n = len(json.load(open(os.path.join(fdir, 'frames.json')))['indices'])
    os.makedirs(a.out, exist_ok=True)

    def run(k):
        r = subprocess.run([onnaq, '-e', 'cpu', a.model, a.qgraph or '-',
                            os.path.join(fdir, 'f%04d.rgb' % k), os.path.join(a.out, 'f%04d' % k)],
                           capture_output=True, text=True)
        if r.returncode:
            raise SystemExit('onnaq failed on frame %d: %s' % (k, r.stderr[-300:]))
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(run, range(n)))
    print('%d frames -> %s' % (n, a.out))


def load_heads(src, n):
    """-> [(h, w, c) float32 arrays N x h x w x c], smallest grid first"""
    if os.path.isdir(src):
        files = sorted({os.path.basename(f).split('.')[1] for f in glob.glob(os.path.join(src, 'f0000.*.f32'))})
        heads = [np.stack([np.fromfile(os.path.join(src, 'f%04d.%s.f32' % (k, j)), np.float32) for k in range(n)])
                 for j in files]
    else:
        heads = [np.fromfile(f, np.float32).reshape(n, -1) for f in sorted(glob.glob(src + '.*.f32'))]
    if not heads:
        raise SystemExit('no head outputs under %s' % src)
    return sorted(heads, key=lambda h: h.shape[1])


def to_outs(heads, n, nc):
    """flat NHWC heads -> {stride: N x C x h x w}, the grid from the 416x320 input"""
    import torch
    c = 3 * (5 + nc)
    outs = {}
    for h in heads:
        cells = h.shape[1] // c
        for s in (64, 32, 16, 8):
            if (H // s) * (W // s) == cells:
                outs[s] = torch.from_numpy(h.reshape(n, H // s, W // s, c)).permute(0, 3, 1, 2)
                break
        else:
            raise SystemExit('head with %d values per frame fits no stride' % h.shape[1])
    return outs


def person_sweep(dets, gts, min_px):
    import torch
    from torchvision.ops import box_iou
    rows = []
    for d, g in zip(dets, gts):
        g = g[g[:, 0] == 0]
        if not (1 <= len(g) <= 2 and (torch.sqrt(g[:, 3] * g[:, 4]) >= min_px).all()):
            continue
        d = d[d[:, 5] == 0]
        gb = torch.stack([g[:, 1] - g[:, 3] / 2, g[:, 2] - g[:, 4] / 2, g[:, 1] + g[:, 3] / 2, g[:, 2] + g[:, 4] / 2], 1)
        if len(d):
            iou = box_iou(gb, d[:, :4])
            best = [float(d[iou[j] >= 0.5, 4].max()) if (iou[j] >= 0.5).any() else 0.0 for j in range(len(g))]
            false = [float(s) for s, m in zip(d[:, 4], (iou >= 0.5).any(0)) if not m]
        else:
            best, false = [0.0] * len(g), []
        rows.append((best, false))
    people = sum(len(b) for b, _ in rows)
    sweep = {}
    for th in THRESHOLDS:
        found = sum(sum(s >= th for s in b) for b, _ in rows)
        fp = sum(sum(s >= th for s in f) for _, f in rows) / max(len(rows), 1)
        sweep['%.2f' % th] = dict(found=found, recall=round(found / max(people, 1), 3), false_per_frame=round(fp, 3))
    return dict(frames=len(rows), people=people, sweep=sweep)


def cmd_score(a):
    from train import decode, ap50, ANCHORS, CLASSES
    ds = dataset(a.data)
    idx = json.load(open(os.path.join(a.frames, 'frames.json')))['indices']
    n = len(idx)
    gts = [ds[i][1][:, 1:] for i in idx]
    anchors = ANCHORS
    if a.anchors:
        v = [int(x) for x in a.anchors.split(',')]
        anchors = {32: list(zip(v[0:6:2], v[1:6:2])), 16: list(zip(v[6:12:2], v[7:12:2]))}
    report = {}
    for src in a.src:
        heads = load_heads(src, n)
        nc = None
        for cand in range(1, 81):
            if all(h.shape[1] % (3 * (5 + cand)) == 0 for h in heads):
                try:
                    to_outs(heads, n, cand)
                    nc = cand
                    break
                except SystemExit:
                    pass
        if nc is None:
            raise SystemExit('%s: cannot tell the class count from the head sizes' % src)
        dets = decode(to_outs(heads, n, nc), conf=0.01, nc=nc, anchors=anchors)
        aps = ap50(dets, gts, nc)
        names = CLASSES[:nc] if len(CLASSES) >= nc else ['class%d' % c for c in range(nc)]
        sw = person_sweep(dets, gts, a.min_px)
        report[src] = dict(frames=n, classes=nc, ap50=dict(zip(names, aps)), person=sw)
        print('%s: %d frames, %d classes, AP50 %s' % (
            src, n, nc, '  '.join('%s %.3f' % (c, v) for c, v in zip(names, aps) if v == v)))
        print('  %d frames with 1-2 people >= %d px (%d people)' % (sw['frames'], a.min_px, sw['people']))
        for th, v in sw['sweep'].items():
            print('  thr %s  found %3d = %.2f   false person boxes/frame %.2f' % (
                th, v['found'], v['recall'], v['false_per_frame']))
    if a.json:
        json.dump(report, open(a.json, 'w'), indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('frames')
    f.add_argument('--data', required=True, help='dataset root (val/images, val/labels)')
    f.add_argument('--n', type=int, default=300)
    f.add_argument('out')
    o = sub.add_parser('onnaq')
    o.add_argument('model')
    o.add_argument('out', help='output folder, inside the frames folder')
    o.add_argument('--qgraph', help='text graph (default: onnaq reads the graph from the .bin)')
    o.add_argument('--onnaq', help='onnaq binary (default tools/onna/onnaq)')
    o.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
    s = sub.add_parser('score')
    s.add_argument('--data', required=True)
    s.add_argument('--min-px', type=int, default=48)
    s.add_argument('--anchors', help='12 ints: stride 32 then stride 16 (w,h pairs)')
    s.add_argument('--json')
    s.add_argument('frames')
    s.add_argument('src', nargs='+')
    a = p.parse_args()
    dict(frames=cmd_frames, onnaq=cmd_onnaq, score=cmd_score)[a.cmd](a)


if __name__ == '__main__':
    main()
