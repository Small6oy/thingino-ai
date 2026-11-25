#!/usr/bin/env python3
"""Score the open pretrained starts on our validation set BEFORE any training (CPU is fine).

    THINGINO_AI_CLASSES=person,pet,vehicle python3 eval_start.py DATA_ROOT [--limit N] [--out FILE.json]
                       [--only yolox,yolox-relu,...] [--calib N]

DATA_ROOT/val/{images,labels} as build_model.py makes it (WORK/dataset). Every variant is scored
with the training code's own AP@0.5 (qat/train.py ap50, 416x320 16:9 centre crop, RGB), so the
numbers compare directly with a build's 'epoch -1' and per-epoch lines.

The ladder shows what each step costs, from the model as Megvii released it to the starting
point of our QAT (what an untrained .bin on the camera would do):

  yolox          YOLOX-Nano as released (SiLU, SPP, Focus stem, its own anchor-free head + decode).
                 Our class = max over its COCO classes (pet = cat/dog, vehicle = car/motorcycle/bus/truck).
  yolox-relu     + the T31 graph changes: ReLU, no SPP pools, bottom/right stride-2 padding,
                 3x3 stem (centre taps). Still YOLOX's own head.
  yoloxn-float   our T31 graph (qat/yoloxn.py from_yolox), float, BatchNorm as released. Our
                 YOLOv5-style anchor heads: box rows start at the anchor, so boxes are not learned.
  yoloxn-bn      the same with BatchNorm statistics re-measured on --calib training images
                 (no training: the ReLU swap moves every layer's statistics).
  yoloxn-int8    yoloxn-bn with BatchNorm folded, activations calibrated, quantisation on: the
                 QAT start, numerically the T31 .bin (bit-exact export).
  mnv4n-float / mnv4n-bn / mnv4n-int8
                 the same for MobileNetV4-conv-small-0.5 + YOLOX-Nano neck/heads (qat/mnv4.py).
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'qat'))
import train as T  # noqa: E402  (reads THINGINO_AI_CLASSES)
import yoloxn  # noqa: E402

VARIANTS = ('yolox', 'yolox-relu', 'yoloxn-float', 'yoloxn-bn', 'yoloxn-int8', 'mnv4n-float', 'mnv4n-bn', 'mnv4n-int8')


def arg(a, k, d):
    return type(d)(a[a.index(k) + 1]) if k in a else d


# ------------------------------------------------------------------ YOLOX's own head
def yolox_decode(levels, conf=0.05):
    """yolox_forward output -> per image (x0, y0, x1, y1, score, cls) in 416x320 pixels."""
    from torchvision.ops import nms
    ids = [yoloxn.COCO.get(c) for c in T.CLASSES]
    allp = []
    for (_, _, cls, reg, obj), s in zip(levels, (8, 16, 32)):
        n, _, gh, gw = reg.shape
        gy, gx = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing='ij')
        cx = (reg[:, 0] + gx) * s; cy = (reg[:, 1] + gy) * s
        bw = reg[:, 2].exp() * s; bh = reg[:, 3].exp() * s
        o = obj[:, 0].sigmoid(); pc = cls.sigmoid()
        sc = torch.stack([pc[:, i].amax(1) if i else torch.zeros_like(o) for i in ids], -1)   # n gh gw C
        score, c = (o[..., None] * sc).max(-1)
        allp.append(torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, score, c.float()], -1).view(n, -1, 6))
    res, out = torch.cat(allp, 1), []
    for d in res:
        d = d[d[:, 4] > conf]
        keep = []
        for c in range(T.NC):
            dc = d[d[:, 5] == c]
            if len(dc):
                keep.append(dc[nms(dc[:, :4], dc[:, 4], 0.45)[:100]])
        out.append(torch.cat(keep) if keep else torch.zeros((0, 6)))
    return out


@torch.no_grad()
def eval_yolox(sd, loader, relu):
    dets, gts = [], []
    stem = yoloxn.first_conv(sd[yoloxn.B + 'stem.conv.conv.weight'], 3) if relu else None
    for x, t in loader:
        if relu:       # T31 changes: input RGB / 255 into our 3x3 stem
            lv = yoloxn.yolox_forward(sd, x / 255.0, act='relu', spp=False, stem=stem, s2pad='same')
        else:          # as released: BGR 0..255
            lv = yoloxn.yolox_forward(sd, x.flip(1), act='silu', spp=True)
        dets += yolox_decode(lv)
        gts += [t[t[:, 0] == i][:, 1:] for i in range(len(x))]
    return T.ap50(dets, gts), dets, gts


# ------------------------------------------------------------------ our graphs
def calib_batches(root, n, bs=16):
    ds = T.DetSet(root, 'train' if os.path.isdir(os.path.join(root, 'train', 'images')) else 'val', False)
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    return [torch.stack([ds[i][0] for i in idx[k:k + bs]]) for k in range(0, len(idx), bs)]


def float_net(arch, root):
    return T.float_start_net(arch, {'yoloxn': 'yolox', 'mnv4n': 'mnv4'}[arch], root).eval()


@torch.no_grad()
def to_int8(net, batches):
    from calib import tame_collapsed_bn, fold_bn, calibrate_acts, fit_requant
    from net import QConv
    tame_collapsed_bn(net)
    fold_bn(net)
    for m in net.modules():
        if isinstance(m, QConv):
            m.bn = None
    calibrate_acts(net, batches)
    fit_requant(net)
    return net.eval()


def main(a):
    if len(a) < 2 or a[1].startswith('-'):
        raise SystemExit(__doc__)
    root = a[1]
    limit = arg(a, '--limit', 0) or None
    only = arg(a, '--only', ','.join(VARIANTS)).split(',')
    out_path = arg(a, '--out', '')
    torch.set_num_threads(arg(a, '--threads', os.cpu_count() or 4))
    va = DataLoader(T.DetSet(root, 'val', False, limit), 16, num_workers=arg(a, '--workers', 4), collate_fn=T.collate)
    print('val %d images, classes %s, variants %s' % (len(va.dataset), ', '.join(T.CLASSES), ', '.join(only)), flush=True)
    batches = None
    res = {}

    def report(name, aps, t0):
        res[name] = dict(ap50=dict(zip(T.CLASSES, [round(float(v), 4) for v in aps])),
                         mean=round(float(np.nanmean(aps)), 4), seconds=round(time.time() - t0),
                         sized=({k: {c: round(float(v), 4) for c, v in d.items()} for k, d in T.LAST_EVAL.items()}
                                if name not in ('yolox', 'yolox-relu') else None))
        print('%-13s AP50 %s  mean %.3f  (%.0f s)' % (name, '  '.join('%s %.3f' % (c, v) for c, v in zip(T.CLASSES, aps)),
                                                    np.nanmean(aps), time.time() - t0), flush=True)
        if out_path:
            json.dump(dict(classes=T.CLASSES, val_images=len(va.dataset), results=res), open(out_path, 'w'), indent=1)

    sd = None
    for name in only:
        t0 = time.time()
        if name in ('yolox', 'yolox-relu'):
            sd = sd or yoloxn.load_state(yoloxn.weights_path())
            aps, dets, gts = eval_yolox(sd, va, name == 'yolox-relu')
            T.LAST_EVAL.clear()
            for k, (lo, hi) in T.SIZE_BUCKETS.items():
                T.LAST_EVAL[k] = dict(zip(T.CLASSES, T.ap50_sized(dets, gts, lo, hi)))
            report(name, aps, t0)
            res[name]['sized'] = {k: {c: round(float(v), 4) for c, v in d.items()} for k, d in T.LAST_EVAL.items()}
            continue
        arch, stage = name.split('-')
        if batches is None:
            batches = calib_batches(root, arg(a, '--calib', 256))
        from calib import bn_stats
        net = float_net(arch, root)
        if stage != 'float':
            bn_stats(net, batches)
        if stage == 'int8':
            to_int8(net, batches)
        report(name, T.evaluate(net, va, 'cpu'), t0)
    if out_path:
        print('wrote %s' % out_path)


if __name__ == '__main__':
    main(sys.argv)
