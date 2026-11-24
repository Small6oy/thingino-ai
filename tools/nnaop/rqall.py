#!/usr/bin/env python3
"""Venus's CPU requant + SiLU for every conv layer, checked against what later layers were fed.

For conv segment L (convref.py must have written seg<L>_meta.npz):
  params at the end of its W block (wend), per 32-oc block k:
     B_k  int32[32]   at wend + 0x25 + 256k
     Fl_k float32[32] at wend + 0xa5 + 256k
  then the SiLU table after b"04 00 00 00 20 00 00 00": 32 segments of 1024 t-units, int16,
     interleaved (dy_i, dy_16+i) x16 then (y_i, y_16+i) x16
  q = rint(PWL(rint(f32(S + B) * Fl)) / 128), all float32  (the layer-0 rule, ref01.py)
Every traced output vector (CPOC x 8 px, the layer's real channels) is requantized and looked up
among the F pixels of the following segments (lane offset 0, or 16 for 16-channel tensors).
usage: rqall.py <dir> [segments...]
"""
import sys

import numpy as np

REC = np.dtype([("op", "<u4"), ("imm", "<u4"), ("vec", "u1", 64)])
ADDS = {5, 12, 15, 22, 25, 28, 35}
MARK = b"\x04\x00\x00\x00\x20\x00\x00\x00"


def params(blob, wend, noc):
    nb = (noc + 31) // 32
    B = np.concatenate([np.frombuffer(blob[wend + 0x25 + 256 * k:wend + 0xa5 + 256 * k], "<i4")
                        for k in range(nb)]).astype(np.int64)
    Fl = np.concatenate([np.frombuffer(blob[wend + 0xa5 + 256 * k:wend + 0x125 + 256 * k], "<f4")
                         for k in range(nb)])
    mk = blob.find(MARK, wend + 0x25 + 256 * nb - 0x80)
    return B, Fl, table32(blob, mk + 8), mk - wend


def table32(blob, o):
    """32 segments of 1024 t-units, stored interleaved: (dy_i, dy_16+i) x16, (y_i, y_16+i) x16"""
    t = np.frombuffer(blob[o:o + 128], "<i2").astype(np.float64)
    dy = np.concatenate([t[0:32:2], t[1:32:2]])
    y = np.concatenate([t[32::2], t[33::2]])
    return dy, y


def requant(S, B, Fl, table):
    dy, ty = table
    t = np.rint((S + B).astype(np.float32) * Fl).astype(np.int64)
    seg = t >> 10
    segc = np.clip(seg, 0, 31)
    f32 = np.float32           # float32 throughout: float64 flips 11 ties in 2.4M (L2, L21)
    y = ty[segc].astype(f32) + dy[segc].astype(f32) * \
        (np.clip(t, 0, 32767) - segc * 1024).astype(f32) / f32(1024)
    y = np.where(seg < 0, f32(ty[0]), y)
    return np.clip(np.rint(y / f32(128)), 0, 255).astype(np.int64), seg


def concat_factors(blob, metas):
    """rescale factors a (F = rint((q - 128) * a) + 128) stored in the .bin, with where they came from:
    - header concat records: n, -128 x n, n, then n input scales and 1/s_out  -> a = s_in / s_out
    - a type-30 record after a conv's tables: a per-channel float repeated over the channels"""
    import struct
    out = []
    for n in range(2, 6):
        pat = struct.pack("<i", n) + struct.pack("<i", -128) * n + struct.pack("<i", n)
        o = blob.find(pat)
        while o >= 0:
            f = struct.unpack("<" + "f" * (n + 1), blob[o + len(pat):o + len(pat) + 4 * (n + 1)])
            out += [(np.float32(f[i]) * np.float32(f[n]), f"header concat @{o} input {i}") for i in range(n)]
            o = blob.find(pat, o + 1)
    for L, (wend, noc) in metas.items():
        B, Fl, tab, mko = params(blob, wend, noc)
        a0 = wend + mko + 8 + 256
        if int.from_bytes(blob[a0:a0 + 4], "little") != 30:
            continue
        for off in range(4):
            h = np.frombuffer(blob[a0 + off:a0 + off + 4000], "<f4")
            vals, cnt = np.unique(h[np.isfinite(h)], return_counts=True)
            for v, c in zip(vals, cnt):
                if c >= 32 and 0.3 < v < 1.5:
                    out.append((v, f"type-30 record after L{L}"))
    return out


def main():
    d = sys.argv[1]
    ops = np.fromfile(f"{d}/nn.bin", dtype=REC)
    r = list(np.load(f"{d}/resets.npy")) + [len(ops)]
    blob = open(f"{d}/yolov5n_t41.bin", "rb").read()
    segs = [int(x) for x in sys.argv[2:]] or [s for s in range(1, len(r) - 1) if s not in ADDS]
    fpix = {}

    def fstream(k):
        if k not in fpix:
            s = ops[r[k]:r[k + 1]]
            fpix[k] = s["vec"][(s["op"] == 2) & ((s["imm"] == 0x20) | (s["imm"] == 0x21))] \
                .reshape(-1, 32)
        return fpix[k]

    tot_found = tot_all = 0
    metas = {}
    for L in range(1, len(r) - 1):
        if L not in ADDS:
            mm = np.load(f"{d}/seg{L}_meta.npz")
            metas[L] = (int(mm["wend"]), int(mm["noc"]))
    factors = concat_factors(blob, metas)
    for L in segs:
        m = np.load(f"{d}/seg{L}_meta.npz")
        name, noc = str(m["name"]), int(m["noc"])
        if "/model.24/" in name:
            print(f"L{L:2} {name}: detection head (float output), skipped")
            continue
        s = ops[r[L]:r[L + 1]]
        v = s["vec"][(s["op"] == 3) & (s["imm"] < 0x10)].copy().view("<i4").reshape(-1, 8, 32)
        tiles = np.load(f"{d}/seg{L}_tiles.npy")
        ob = m["cpoc_ob"]
        B, Fl, table, mko = params(blob, int(m["wend"]), noc)
        n = min(noc, 32)
        # per CPOC: its block's 32 params
        Bv = B.reshape(-1, 32)[ob][:, None, :n]
        Fv = Fl.reshape(-1, 32)[ob][:, None, :n]
        q, sg = requant(v[:, :, :n].astype(np.int64), Bv, Fv, table)
        valid = np.array([[(pm >> p) & 1 for p in range(8)] for pm in tiles[:, 5]], bool)
        # consumer slices of the next segments
        look = set()
        for k in range(L + 1, min(L + 31, len(r) - 1)):
            f = fstream(k)
            for o in ((0, 16) if n == 16 else (0,)):
                look.update(map(bytes, np.ascontiguousarray(f[:, o:o + n]).astype(np.uint8)))
        qb = q.astype(np.uint8)
        found = np.zeros(q.shape[:2], bool)
        for i in range(q.shape[0]):
            for p in range(8):
                if valid[i, p]:
                    found[i, p] = qb[i, p].tobytes() in look
        nv = int(valid.sum())
        nf = int(found.sum())
        note = ""
        if nf < nv:            # a concat input: try the stored rescale factors
            for a, src in factors:
                g = np.clip(np.rint((q.astype(np.float32) - 128) * np.float32(a)) + 128, 0, 255) \
                    .astype(np.uint8)
                f2 = np.array([[valid[i, p] and g[i, p].tobytes() in look for p in range(8)]
                               for i in range(len(g))])
                if f2.sum() == nv:
                    nf, note = nv, f"  [concat rescale a={a:.7f}, {src}]"
                    break
        past = int(((sg[valid] > 31) | (sg[valid] < 0)).any(-1).sum())
        tot_found += nf
        tot_all += nv
        print(f"L{L:2} {name:32} oc {noc:3} table@+{mko:4}: requantized vectors found downstream"
              f" {nf}/{nv} ({100 * nf / max(nv, 1):.2f}%); vectors with a lane outside 0..32767 {past}{note}",
              flush=True)
    print(f"total: {tot_found}/{tot_all} ({100 * tot_found / max(tot_all, 1):.3f}%)")


if __name__ == "__main__":
    main()
