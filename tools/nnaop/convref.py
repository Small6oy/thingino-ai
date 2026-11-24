#!/usr/bin/env python3
"""NumPy reference of one NNA conv layer, checked against the trace.

usage: convref.py <dir> <segment> [onnx conv index]
  - W: the layer's .bin block (the unique 256-B entries of its NNDWR stream, found in the file),
    decoded to groups Weff[g][oc][ic] (8-bit, msf 2), each group matched to an ONNX tap
    (ky, kx) and input-channel plane;
  - input: the layer's traced F bytes placed by fmap.py (seg<N>_fmap.npy), pad 128 (reg 0x2f);
  - output: sums vs every NNDRD 0x00-0x0f read (raster order, 8 px x 32 oc per CPOC).
"""
import sys

import numpy as np
import onnx
from onnx import numpy_helper

REC = np.dtype([("op", "<u4"), ("imm", "<u4"), ("vec", "u1", 64)])


def wblock(ps, blob):
    """the .bin offset of the layer's W block and its entries in file order"""
    ents = [b"".join(ps[i:i + 4]) for i in range(0, len(ps), 4)]
    ue = list(dict.fromkeys(ents))
    offs = sorted(blob.find(e) for e in ue)
    if offs[0] < 0 or any(b - a != 256 for a, b in zip(offs, offs[1:])):
        return None, None
    o = offs[0]
    raw = np.frombuffer(blob[o:o + 256 * len(ue)], np.uint8).reshape(-1, 256)
    return o, raw


def groups(raw, wbit=8):
    sl = wbit // 2
    G = np.zeros((len(raw) // sl, 32, 32), np.int64)
    for g in range(len(G)):
        for k in range(sl):
            e = raw[g * sl + k].reshape(32, 8).astype(np.int64)
            for ic in range(32):
                G[g, :, ic] |= ((e[:, ic // 4] >> (2 * (ic % 4))) & 3) << (2 * k)
    return G - (1 << (wbit - 1))


def onnx_convs(path):
    m = onnx.load(path)
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    out = []
    for n in m.graph.node:
        if n.op_type != "Conv":
            continue
        at = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        w = init[n.input[1]].astype(np.float64)
        s = np.abs(w).reshape(w.shape[0], -1).max(1) / 127
        q = np.round(w / s[:, None, None, None]).astype(np.int64)
        out.append(dict(name=n.name, q=q, stride=at["strides"][0], pad=at["pads"][0]))
    return out


def main():
    d, seg = sys.argv[1], int(sys.argv[2])
    ops = np.fromfile(f"{d}/nn.bin", dtype=REC)
    r = list(np.load(f"{d}/resets.npy")) + [len(ops)]
    s = ops[r[seg]:r[seg + 1]]
    blob = open(f"{d}/yolov5n_t41.bin", "rb").read()
    ps = [bytes(v) for v in s["vec"][(s["op"] == 2) & (s["imm"] < 0x20)]]
    off, raw = wblock(ps, blob)
    G = groups(raw)
    print(f"segment {seg}: W block at .bin {off}, {len(raw)} entries -> {len(G)} groups")

    # what the engine did with each group: (input plane, ky, kx), from fmap.py's replay
    ents = [b"".join(ps[i:i + 4]) for i in range(0, len(ps), 4)]
    ent_file = [(blob.find(e) - off) // 256 for e in ents]
    sl = 4
    use = {}
    for ent, pl, ky, kx in np.load(f"{d}/seg{seg}_wuse.npy"):
        g = ent_file[ent] // sl
        use.setdefault(g, set()).add((int(pl), int(ky), int(kx)))
    multi = [g for g, u in use.items() if len(u) > 1]
    print(f"   groups used: {len(use)}/{len(G)}; used with more than one (plane, tap): {len(multi)}")

    convs = onnx_convs(f"{d}/yolov5n.onnx")
    cands = [int(sys.argv[3])] if len(sys.argv) > 3 else range(len(convs))
    best = None
    for ci in cands:
        q = convs[ci]["q"]
        noc, nic, kh, kw = q.shape
        if (noc + 31) // 32 * kh * kw * ((nic + 31) // 32) > len(G) * 2 or \
                any(ky >= kh or kx >= kw for u in use.values() for _, ky, kx in u):
            continue
        # every group: its tap is known; find oc block and the ONNX input channels of its columns
        mp, err = {}, 0
        for g, u in use.items():
            pl, ky, kx = next(iter(u))
            cols = np.flatnonzero(np.abs(G[g, :min(noc, 32)]).sum(0) > 0)   # real oc rows only
            k = int(cols.max()) + 1 if len(cols) else 1
            bestg = None
            for ob in range((noc + 31) // 32):
                ref_all = q[32 * ob:32 * ob + 32, :, ky, kx]
                for o in range(0, max(nic - k, 0) + 1):
                    ref = ref_all[:, o:o + k]
                    if ref.shape[1] != k:
                        continue
                    e = int(np.abs(G[g, :ref.shape[0], :k] - ref).max())
                    if bestg is None or e < bestg[0]:
                        bestg = (e, ob, o)
            if bestg is None:
                bestg = (999, 0, 0)
            mp[g] = (bestg[1], pl, ky, kx, bestg[2], k)
            err = max(err, bestg[0])
        if best is None or err < best[0]:
            best = (err, ci, mp)
    err, ci, mp = best
    conv = convs[ci]
    q = conv["q"]
    noc, nic, kh, kw = q.shape
    print(f"ONNX {conv['name']} {q.shape} stride {conv['stride']} pad {conv['pad']}:"
          f" every group within ±{err} of its tap")
    planes = {}
    for g, (ob, pl, ky, kx, o, k) in mp.items():
        planes.setdefault(pl, set()).add((o, k))
    print("   input plane -> ONNX channels (first, count):",
          {pl: sorted(v) for pl, v in sorted(planes.items())})

    # ---- check 1 (any layer): every traced sum = sum of W group x F pixel over what the MACs read
    f = s["vec"][(s["op"] == 2) & ((s["imm"] == 0x20) | (s["imm"] == 0x21))].reshape(-1, 32)
    v = s["vec"][(s["op"] == 3) & (s["imm"] < 0x10)].copy().view("<i4").reshape(-1, 8, 32)
    pr = np.load(f"{d}/seg{seg}_pairs.npy")
    fx = np.concatenate([f.astype(np.int64), np.full((1, 32), 128, np.int64)])   # last row = pad
    src = np.where(pr[:, 3] < 0, len(f), pr[:, 3])
    gi = np.array([ent_file[e] // sl if e >= 0 else -1 for e in pr[:, 2]])
    contrib = np.einsum("noc,nc->no", G[gi], fx[src])
    acc = np.zeros((len(v) * 8, 32), np.int64)
    np.add.at(acc, pr[:, 0] * 8 + pr[:, 1], contrib)
    acc = acc.reshape(-1, 8, 32)
    n = min(len(acc), len(v))
    eq = (acc[:n] == v[:n]).all(-1)
    print(f"check 1: {len(v)} CPOC x 8 px traced, {len(pr)} MAC products replayed;"
          f" sums equal {int(eq.sum())}/{eq.size}; bit-exact {bool(eq.all()) and len(acc) == len(v)}")
    sys.stdout.flush()
    tiles = np.load(f"{d}/seg{seg}_tiles.npy")
    cpoc_ob = np.array([mp[ent_file[w] // sl][0] if w >= 0 and ent_file[w] // sl in mp else -1
                        for w in tiles[:, 6]], np.int64)
    np.savez(f"{d}/seg{seg}_meta.npz", ci=ci, name=conv["name"], wend=off + 256 * len(raw),
             noc=noc, cpoc_ob=cpoc_ob)
    if not (convs[ci]["q"].shape[2] > 1):
        return          # 1x1: venus tiles them locally; no image positions to check
    print("check 2 (positions, 3x3 only):")
    # input map from the traced F stream
    f = s["vec"][(s["op"] == 2) & ((s["imm"] == 0x20) | (s["imm"] == 0x21))].reshape(-1, 32)
    m = np.load(f"{d}/seg{seg}_fmap.npy")
    ok = m[:, 0] >= 0
    H, W = int(m[ok, 0].max()) + 1, int(m[ok, 1].max()) + 1
    P = int(m[ok, 2].max()) + 1
    X = np.full((H, W, 32 * P), -1, np.int64)
    for (y, x, pl), v in zip(m[ok], f[ok]):
        X[y, x, 32 * pl:32 * pl + 32] = v
    print(f"input {H}x{W}x{32 * P} from the F stream; unfilled bytes: {int((X < 0).sum())}")

    st, pd = conv["stride"], conv["pad"]
    OH, OW = (H + 2 * pd - kh) // st + 1, (W + 2 * pd - kw) // st + 1
    Xp = np.full((H + 2 * pd + st, W + 2 * pd + st, 32 * P), 128, np.int64)   # pad 128 (0x2f)
    Xp[pd:pd + H, pd:pd + W] = X
    NOB = (noc + 31) // 32
    Sm = np.zeros((OH, OW, 32 * NOB), np.int64)
    for g, (ob, pl, ky, kx, _, _) in mp.items():
        patch = Xp[ky:ky + st * OH:st, kx:kx + st * OW:st, 32 * pl:32 * pl + 32]
        Sm[..., 32 * ob:32 * ob + 32] += np.einsum("hwc,oc->hwo", patch, G[g])

    v = s["vec"][(s["op"] == 3) & (s["imm"] < 0x10)].copy().view("<i4").reshape(-1, 8, 32)
    tiles = np.load(f"{d}/seg{seg}_tiles.npy")
    print(f"NNDRD: {len(v)} groups of 8 px x 32 oc, CPOC tiles {len(tiles)};"
          f" output {OH}x{OW}x{noc} = {OH * OW * NOB} px-blocks")
    # streamed W entry -> file entry -> group -> oc block
    T = np.full((OH, OW, 32 * NOB), np.iinfo(np.int64).min, np.int64)
    placed = clash = 0
    for k, (fx, fy, rows2, sx, sy, pmask, went) in enumerate(tiles):
        ob = mp[ent_file[went] // sl][0] if went >= 0 and ent_file[went] // sl in mp else 0
        for px in range(8):
            if not pmask & (1 << px):
                continue
            rr, cc = (px >> 2, px & 3) if rows2 else (0, px)
            iy, ix = fy + rr * sy + pd, fx + cc * sx + pd
            if iy % st or ix % st:
                continue
            oy, ox = iy // st, ix // st
            if not (0 <= oy < OH and 0 <= ox < OW):
                continue
            blk = T[oy, ox, 32 * ob:32 * ob + 32]
            if blk[0] != np.iinfo(np.int64).min and not (blk == v[k, px]).all():
                clash += 1
            blk[:] = v[k, px]
            placed += 1
    filled = (T != np.iinfo(np.int64).min).all(-1)
    eq = (T == Sm).all(-1)
    print(f"  placed {placed} px-blocks by tile registers ({clash} clashes);"
          f" outputs filled {int(filled.sum())}/{OH * OW}")
    print(f"  sums equal to NumPy: {int(eq.sum())}/{OH * OW} pixels; bit-exact {np.array_equal(T, Sm)}")
    if np.array_equal(T, Sm):
        np.save(f"{d}/seg{seg}_sums.npy", T)
    else:
        bad = np.argwhere(~eq)[:5]
        for oy, ox in bad:
            print("   differ at", oy, ox, "trace", T[oy, ox, :4], "numpy", Sm[oy, ox, :4])

if __name__ == "__main__":
    main()
