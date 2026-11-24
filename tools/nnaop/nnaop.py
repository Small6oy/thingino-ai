#!/usr/bin/env python3
"""Decode a T41 NNA trace from gtxaspec QEMU (ingenic-nna).

Inputs (one emulator run):
  nn.bin     -global ingenic-nna.log=...: per NN instruction <u32 op><u32 imm><64 B vec>
  trace.log  -trace ingenic_nna_write -trace ingenic_nna_op -D ...: register/descriptor
             writes interleaved with the NN ops (gives the DMA order)

Writes a summary (command types, layers, DMA) to stdout and events.npz for later steps.
Op numbering and semantics follow hw/misc/ingenic-nna.c (QEMU ingenic-0.7.26).
"""
import re
import sys
from collections import Counter, defaultdict

import numpy as np

OPS = ["NNRWR", "NNRRD", "NNDWR", "NNDRD", "NNMAC", "NNCMD"]
REC = np.dtype([("op", "<u4"), ("imm", "<u4"), ("vec", "u1", 64)])
CHAN_TO_ORAM = {0: 1, 1: 1, 2: 1, 4: 0, 5: 0}   # +0x8000 + 4*n


def load_ops(path):
    return np.fromfile(path, dtype=REC)


def parse_trace(path):
    """-> list of DMA chains (op_index_before, channel, [(ddr, oram_off, bytes)])"""
    win = {}
    dmas = []
    nop = 0
    rw = re.compile(r"ingenic_nna_write \+0x([0-9a-f]+) size (\d) value 0x([0-9a-f]+)")
    with open(path) as f:
        for line in f:
            if line.startswith("ingenic_nna_op"):
                nop += 1
                continue
            m = rw.match(line)
            if not m:
                continue
            off, size, val = int(m[1], 16), int(m[2]), int(m[3], 16)
            if off >= 0x8000:
                ch = (off - 0x8000) // 4
                if val & (1 << 31):
                    idx = val & 2047
                    chain = []
                    for _ in range(2048):
                        lo = win.get(idx * 8, 0)
                        hi = win.get(idx * 8 + 4, 0)
                        n = ((lo >> 1) & 0x1f) << 12 | (hi >> 20)
                        chain.append((lo & ~0x3f, (hi & 0xfffff) * 64, (n + 1) * 64))
                        idx = (idx + 1) & 2047
                        if not lo & 1:
                            break
                    dmas.append((nop, ch, chain))
            else:
                assert size == 4, line
                win[off] = val
    return dmas, nop


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    ops = load_ops(f"{d}/nn.bin")
    dmas, ntrace = parse_trace(f"{d}/trace.log")
    print(f"NN ops: {len(ops)} (trace counted {ntrace})")
    assert len(ops) == ntrace

    op, imm = ops["op"], ops["imm"]
    word0 = ops["vec"][:, :4].copy().view("<u4")[:, 0]

    # ---- command types
    print("\n== command types (op, imm) ==")
    for o in range(6):
        sel = op == o
        c = Counter(imm[sel].tolist())
        if o == 0:
            print(f"{OPS[o]:6} {sel.sum():8}  registers: " +
                  " ".join(f"{r:#04x}:{n}" for r, n in sorted(c.items())))
        else:
            print(f"{OPS[o]:6} {sel.sum():8}  imm: " +
                  " ".join(f"{r:#04x}:{n}" for r, n in sorted(c.items())))

    # ---- segments: NNCMD 0 resets the engine
    resets = np.flatnonzero((op == 5) & (imm == 0))
    print(f"\n== engine resets (NNCMD 0): {len(resets)} ==")

    # replay register file per segment for a config summary
    keys = {0x3b: "fmt", 0x3e: "taps", 0x3c: "planes", 0x46: "msf", 0x44: "obit",
            0x3a: "stride", 0x38: "mactl", 0x55: "vec", 0x2c: "fh", 0x2e: "fw",
            0x37: "plpx", 0x39: "rowpx", 0x43: "qsh", 0x3f: "pxmask", 0x5a: "wfcwn"}
    bounds = list(resets) + [len(ops)]
    dma_at = np.array([x[0] for x in dmas])
    segs = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        so, si = op[a:b], imm[a:b]
        cfg = defaultdict(set)
        for j in np.flatnonzero(so == 0):
            r = int(si[j])
            if r in keys:
                cfg[keys[r]].add(int(word0[a + j]))
        nd = ((dma_at >= a) & (dma_at < b)).sum()
        segs.append(dict(a=a, b=b,
                         mac=int((so == 4).sum()),
                         wq=int(((so == 2) & (si < 0x20)).sum()),
                         bt=int(((so == 2) & (si == 0x40)).sum()),
                         f=int(((so == 2) & ((si == 0x20) | (si == 0x21))).sum()),
                         qsum=int(((so == 5) & ((si & 8) != 0)).sum()),
                         drd=int((so == 3).sum()), dma=int(nd), cfg=cfg))
    print(f"{'seg':>4} {'first op':>9} {'ops':>7} {'MAC':>6} {'Wq':>6} {'BT':>4} {'F':>6} "
          f"{'QSUM':>6} {'DRD':>6} {'DMA':>4}  config")
    for i, s in enumerate(segs):
        cfg = " ".join(f"{k}={'/'.join(hex(v) for v in sorted(vs)[:4])}"
                       for k, vs in sorted(s["cfg"].items()))
        print(f"{i:4} {s['a']:9} {s['b'] - s['a']:7} {s['mac']:6} {s['wq']:6} {s['bt']:4} "
              f"{s['f']:6} {s['qsum']:6} {s['drd']:6} {s['dma']:4}  {cfg}")

    # ---- DMA summary
    print(f"\n== NNDMA: {len(dmas)} chains ==")
    per = Counter()
    for _, ch, chain in dmas:
        per[ch] += 1
    for ch, n in sorted(per.items()):
        tot = sum(sum(c[2] for c in chain) for _, c2, chain in dmas if c2 == ch)
        print(f"  +0x{0x8000 + 4 * ch:x} ({'DDR->ORAM' if CHAN_TO_ORAM[ch] else 'ORAM->DDR'}): "
              f"{n} chains, {tot} bytes")
    flat = [(n, ch, *c) for n, ch, chain in dmas for c in chain]
    np.save(f"{d}/dma.npy", np.array(flat, dtype=np.int64))
    np.save(f"{d}/resets.npy", resets)


if __name__ == "__main__":
    main()
