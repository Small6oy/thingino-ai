#!/usr/bin/env python3
"""Map one layer's NNA weights: NNDWR W writes -> Weff[oc][ic] groups -> ONNX conv -> .bin.

usage: wmap.py <dir with nn.bin, resets.npy, dma.npy> <model.bin> <conv_q.npy> [segment]
W entry layout (hw/misc/ingenic-nna.c nn_wgroup): 256 B = [32 oc][8 B = 32 ic x 2 bit, LSB first];
a wbit-bit weight is wbit/2 entries (2-bit slices, slice k in entry k); msf_mode 2: w - 2^(wbit-1).
"""
import sys

import numpy as np

REC = np.dtype([("op", "<u4"), ("imm", "<u4"), ("vec", "u1", 64)])


def main():
    d, model, convq = sys.argv[1:4]
    seg = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    ops = np.fromfile(f"{d}/nn.bin", dtype=REC)
    resets = np.load(f"{d}/resets.npy")
    bounds = list(resets) + [len(ops)]
    a, b = bounds[seg], bounds[seg + 1]
    s = ops[a:b]
    w0 = s["vec"][:, :4].copy().view("<u4")[:, 0]

    # replay the W write pointer (0x5c) and the format registers
    wram = {}
    wr_w = 0
    fmt = msf = None
    order = []          # (entry, quarter) in write order
    payloads = []
    for i in range(len(s)):
        op, imm = int(s["op"][i]), int(s["imm"][i])
        if op == 0 and imm == 0x5c:
            wr_w = int(w0[i]) & 0xffff
        elif op == 0 and imm == 0x3b:
            fmt = int(w0[i]) & 0xff
        elif op == 0 and imm == 0x46:
            msf = int(w0[i]) & 3
        elif op == 2 and imm < 0x20:
            q = (imm >> 1) & 3
            wram.setdefault(wr_w, bytearray(256))[q * 64:(q + 1) * 64] = bytes(s["vec"][i])
            order.append((wr_w, q))
            payloads.append(bytes(s["vec"][i]))
            wr_w += imm & 1
    wbit = ((fmt >> 3) & 7) + 2
    sl = wbit // 2
    ents = sorted(wram)
    print(f"segment {seg}: ops {a}..{b}, fmt {fmt:#x} (wbit {wbit}, ifbit {(fmt & 7) * 2 + 2}),"
          f" msf {msf}, W entries {ents[0]}..{ents[-1]} ({len(ents)}), {len(payloads)} NNDWR")

    groups = []
    for g in range(len(ents) // sl):
        e0 = ents[g * sl]
        v = np.zeros((32, 32), np.int64)
        for k in range(sl):
            ent = np.frombuffer(bytes(wram[e0 + k]), np.uint8).reshape(32, 8)
            for ic in range(32):
                v[:, ic] |= ((ent[:, ic // 4] >> (2 * (ic % 4))) & 3).astype(np.int64) << (2 * k)
        w = {0: v, 1: 2 * v, 2: v - (1 << (wbit - 1)), 3: 2 * v - ((1 << wbit) - 1)}[msf]
        groups.append(w)
    G = np.stack(groups)                       # [g][oc][ic]
    print("groups", G.shape, "nonzero oc rows:", int((np.abs(G).sum((0, 2)) > 0).sum()),
          "nonzero ic cols:", int((np.abs(G).sum((0, 1)) > 0).sum()))
    np.save(f"{d}/seg{seg}_weff.npy", G)

    # match every (group, ic) column against the ONNX conv's (c, ky, kx) columns
    q = np.load(convq).astype(np.int64)        # [oc][c][ky][kx]
    noc = q.shape[0]
    cols = q.reshape(noc, -1)                  # [oc][c*ky*kx]
    names = [(c, ky, kx) for c in range(q.shape[1]) for ky in range(q.shape[2])
             for kx in range(q.shape[3])]
    exact = near = zero = 0
    table = {}
    for g in range(G.shape[0]):
        for ic in range(32):
            col = G[g, :noc, ic]
            if not col.any():
                zero += 1
                continue
            hit = np.flatnonzero((cols == col[:, None]).all(0))
            if len(hit):
                exact += 1
                table[(g, ic)] = [names[h] for h in hit]
                continue
            diff = np.abs(cols - col[:, None]).max(0)
            h = int(diff.argmin())
            table[(g, ic)] = [("~", *names[h], int(diff[h]))]
            near += 1
    print(f"ONNX match: {exact} exact, {near} nearest-only, {zero} zero columns;"
          f" oc rows beyond {noc}: {'all zero' if not G[:, noc:].any() else 'NONZERO'}")
    for g in range(G.shape[0]):
        print(f" g{g}: " + " ".join(
            f"{ic}={'/'.join(','.join(map(str, t)) for t in table[(g, ic)][:2])}"
            for ic in range(32) if (g, ic) in table))

    # where do the NNDWR payloads sit in the model file?
    blob = open(model, "rb").read()
    offs = []
    for p in payloads:
        offs.append(blob.find(p) if any(p) else -2)
    found = [o for o in offs if o >= 0]
    print(f"\n.bin: {len(found)}/{sum(1 for o in offs if o != -2)} non-zero payloads found;"
          f" offsets {min(found) if found else None}..{max(found) if found else None}")
    if found:
        o = np.array(offs)
        st = np.diff(o[o >= 0])
        print("  stride between consecutive payloads:", dict(zip(*np.unique(st, return_counts=True))))
    # DMA chunks to the ORAM within this segment
    dma = np.load(f"{d}/dma.npy")              # (op_idx, ch, ddr, oram_off, bytes)
    sel = dma[(dma[:, 0] >= a) & (dma[:, 0] < b)]
    print(f"  DMA chunks in segment: {len(sel)};"
          f" DDR->ORAM {int(sel[sel[:, 1] < 3][:, 4].sum())} B, ORAM->DDR {int(sel[sel[:, 1] >= 3][:, 4].sum())} B")
    pre = dma[dma[:, 0] < a]
    if len(pre):
        print(f"  DMA chunks before segment: {len(pre)}")
    for r in sel[:12]:
        print(f"   op+{int(r[0]) - a:6} ch {int(r[1])} ddr {int(r[2]):#010x} oram +{int(r[3]):#07x} {int(r[4])} B")


if __name__ == "__main__":
    main()
