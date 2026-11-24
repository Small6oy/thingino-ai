#!/usr/bin/env python3
"""NumPy reference of yolov5n layer 0 on the T41 (venus 0.1.2.5 + NNA), checked against a trace.

  image (AIP RGBA 640x384) --NNA conv 6x6/2, W from the .bin--> int32 sums [192][320][32]
       --CPU requant + SiLU (constants from the .bin)--> layer 1 input bytes [192][320][32]

Checks: (1) the sums against every NNDRD read of layer 0 (raster order), (2) the bytes against
every F pixel layer 1 writes to the NNA (positions from fmap.py's replay).

usage: ref01.py <dir>   needs nn.bin, resets.npy, seg1_fmap.npy (fmap.py <dir> 1),
                        input_rgba.raw, yolov5n_t41.bin

.bin offsets are for our yolov5n_t41.bin (found by search, not parsed from its header yet):
  W block 43369 (24 entries x 256 B), B int32[32] 0xc18e, Fl float32[16] 0xc20e,
  SiLU table int16 0xc2f0: 32 segments, (dy_i, dy_16+i) x16 then (y_i, y_16+i) x16.
"""
import sys

import numpy as np

REC = np.dtype([("op", "<u4"), ("imm", "<u4"), ("vec", "u1", 64)])
H, W = 384, 640
SEG = 1024                       # t units per table segment


def load_layer0(blob):
    wraw = np.frombuffer(blob[43369:43369 + 6144], np.uint8).reshape(24, 256)
    B = np.frombuffer(blob[0xc18e:0xc18e + 128], "<i4").astype(np.int64)
    Fl = np.frombuffer(blob[0xc20e:0xc20e + 64], "<f4")
    t = np.frombuffer(blob[0xc2f0:0xc2f0 + 128], "<i2").astype(np.float64)
    # 32 segments of 1024, stored interleaved: (dy_i, dy_16+i) x16 then (y_i, y_16+i) x16
    return wraw, B, Fl, (np.concatenate([t[0:32:2], t[1:32:2]]), np.concatenate([t[32::2], t[33::2]]))


def decode_w(wraw, wbit=8):
    """W entries -> Weff [group][oc][ic], msf_mode 2 (w = v - 2^(wbit-1))"""
    sl = wbit // 2
    G = np.zeros((len(wraw) // sl, 32, 32), np.int64)
    for g in range(len(G)):
        for k in range(sl):
            e = wraw[g * sl + k].reshape(32, 8).astype(np.int64)
            for ic in range(32):
                G[g, :, ic] |= ((e[:, ic // 4] >> (2 * (ic % 4))) & 3) << (2 * k)
    return G - (1 << (wbit - 1))


def nna_conv0(img, G):
    """the stem: group = kx, ic = 4*ky + c; 6x6, stride 2, pad 2 -> sums [192][320][32]"""
    wq = np.zeros((32, 4, 6, 6), np.int64)
    for kx in range(6):
        for ky in range(6):
            for c in range(4):
                wq[:, c, ky, kx] = G[kx, :, 4 * ky + c]
    x = np.zeros((H + 4, W + 4, 4), np.int64)
    x[2:H + 2, 2:W + 2] = img
    S = np.zeros((H // 2, W // 2, 32), np.int64)
    for ky in range(6):
        for kx in range(6):
            S += np.einsum("hwc,oc->hwo", x[ky:ky + H:2, kx:kx + W:2], wq[:, :, ky, kx])
    return S


def silu(z):
    return z / (1 + np.exp(-z))


def cpu_requant(S, B, Fl, table):
    """t = rint(float32(S + B) * Fl); PWL SiLU over 32 segments of 1024; q = rint(y / 128)"""
    dy, ty = table
    n = len(Fl)
    t = np.rint((S[..., :n] + B[:n]).astype(np.float32) * Fl).astype(np.int64)
    seg = t >> 10
    segc = np.clip(seg, 0, 31)
    f32 = np.float32           # float32 interpolation (float64 flips rare ties on later layers)
    y = ty[segc].astype(f32) + dy[segc].astype(f32) * \
        (np.clip(t, 0, 32767) - segc * SEG).astype(f32) / f32(SEG)
    y = np.where(seg < 0, f32(ty[0]), y)
    q = np.clip(np.rint(y / f32(128)), 0, 255).astype(np.int64)
    out = np.full(S.shape, 128, np.int64)        # channels without parameters: B = 0 -> 128
    out[..., :n] = q
    return out, seg


def main():
    d = sys.argv[1]
    blob = open(f"{d}/yolov5n_t41.bin", "rb").read()
    ops = np.fromfile(f"{d}/nn.bin", dtype=REC)
    r = np.load(f"{d}/resets.npy")
    img = np.fromfile(f"{d}/input_rgba.raw", np.uint8).reshape(H, W, 4)
    wraw, B, Fl, table = load_layer0(blob)

    S = nna_conv0(img, decode_w(wraw))
    s0 = ops[r[0]:r[1]]
    drd = s0["vec"][(s0["op"] == 3) & (s0["imm"] < 0x10)].copy().view("<i4").reshape(S.shape)
    print(f"(1) NNA sums: NumPy == trace for {int((drd == S).all(-1).sum())}/{S.shape[0] * S.shape[1]}"
          f" pixels (32 oc each); bit-exact: {np.array_equal(drd, S)}")

    Q, seg = cpu_requant(S, B, Fl, table)
    s1 = ops[r[1]:r[2]]
    f = s1["vec"][(s1["op"] == 2) & ((s1["imm"] == 0x20) | (s1["imm"] == 0x21))].reshape(-1, 32)
    m = np.load(f"{d}/seg1_fmap.npy")
    ok = m[:, 0] >= 0
    got = Q[m[ok, 0], m[ok, 1]]
    want = f[ok].astype(np.int64)
    bad = got != want
    sg = seg[m[ok, 0], m[ok, 1]]
    tot = bad.size
    print(f"(2) layer 1 input bytes: {tot - int(bad.sum())}/{tot} equal "
          f"({100 * (1 - bad.mean()):.4f}%) over {int(ok.sum())} F pixels x 32 ch")
    for nm, sel in (("seg < 0 (y = y[0])", sg < 0), ("seg 0-15", (sg >= 0) & (sg <= 15)),
                    ("seg 16-31", (sg >= 16) & (sg <= 31)), ("seg > 31", sg > 31)):
        print(f"    {nm:18}: {int(bad[:, :16][sel].sum())} wrong of {int(sel.sum())}")
    print(f"    channels 16-31 (no parameters, 128): {int(bad[:, 16:].sum())} wrong")


if __name__ == "__main__":
    main()
