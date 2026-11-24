#!/usr/bin/env python3
"""The residual adds (C3 shortcuts) on the T41 NNA: a NumPy reference checked against the trace.

venus runs a + b as a 2-tap conv: taps (0,0) and (1,0) over an input whose rows interleave the
operands (row 2y = a, row 2y+1 = b; stride y 2), diagonal unsigned weights wA, wB (msf 0), and
QSUM requant in hardware with one bias B and divisor T for all channels:
    out = clip(floor((64 * (wA*a + wB*b) + B) / T) + 128, 0, 255),   B = T/2 - 8192*(wA + wB)
(64 = QSUM shift 5 for 0x43 = 1, +1 for msf 0; obits 8.)
usage: addref.py <dir> [segments]   (fmap.py <dir> <seg> first)
"""
import sys

import numpy as np

REC = np.dtype([("op", "<u4"), ("imm", "<u4"), ("vec", "u1", 64)])
ADDS = [5, 12, 15, 22, 25, 28, 35]


def wdiag(entries):
    """an 8-bit diagonal weight from 4 W entries (2-bit slices, entry k = slice k)"""
    v = 0
    for k, e in enumerate(entries):
        ent = e.reshape(32, 8)
        sl = [(ent[i, i // 4] >> (2 * (i % 4))) & 3 for i in range(32)]
        assert len(set(sl)) == 1, "not a uniform diagonal"
        v |= int(sl[0]) << (2 * k)
    return v


def main():
    d = sys.argv[1]
    segs = [int(x) for x in sys.argv[2:]] or ADDS
    ops = np.fromfile(f"{d}/nn.bin", dtype=REC)
    r = list(np.load(f"{d}/resets.npy")) + [len(ops)]
    for seg in segs:
        s = ops[r[seg]:r[seg + 1]]
        W = s["vec"][(s["op"] == 2) & (s["imm"] < 0x20)].reshape(-1, 256)
        wA, wB = wdiag(W[0:4]), wdiag(W[4:8])
        bt = s["vec"][(s["op"] == 2) & (s["imm"] == 0x40)].copy().view("<i4").reshape(-1)
        Bv, Tv = bt[:32], bt[32:64] & 0x3ffffff
        assert len(set(Bv)) == 1 and len(set(Tv)) == 1
        B, T = int(Bv[0]), int(Tv[0])
        # operands per output pixel, from the replay: the two (W entry, F pixel) products of each
        # CPOC pixel (entries 0-3 = group A, 4-7 = group B)
        f = s["vec"][(s["op"] == 2) & ((s["imm"] == 0x20) | (s["imm"] == 0x21))].reshape(-1, 32)
        f = f.astype(np.int64)
        pr = np.load(f"{d}/seg{seg}_pairs.npy")          # (cpoc, px, streamed W entry, F src)
        out = s["vec"][(s["op"] == 3) & (s["imm"] == 0x20)].reshape(-1, 8, 32).astype(np.int64)
        A = np.full((len(out), 8, 32), -1, np.int64)
        Bop = np.full((len(out), 8, 32), -1, np.int64)
        for k, px, ent, src in pr:
            if k >= len(out) or src < 0:
                continue
            (A if ent < 4 else Bop)[k, px] = f[src]
        have = (A[..., 0] >= 0) & (Bop[..., 0] >= 0)
        ref = np.clip(np.floor_divide(64 * (wA * A + wB * Bop) + B, T) + 128, 0, 255)
        eq = (ref == out).all(-1) & have
        print(f"seg {seg}: wA {wA} wB {wB} B {B} T {T} (= T/2 - 8192(wA+wB): "
              f"{B == T // 2 - 8192 * (wA + wB)}); outputs {len(out)} x 8 px, with both operands"
              f" {int(have.sum())}; equal {int(eq.sum())}/{int(have.sum())}")

if __name__ == "__main__":
    main()
