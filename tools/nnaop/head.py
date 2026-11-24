#!/usr/bin/env python3
"""The detection heads (YOLOv5 /model.24/m.*): float output of the 1x1 convs, no SiLU.

  out = float32(S + B) * Fl        (B int32, Fl float32 per oc: the same record as every conv)
checked against the head outputs venus returned (NNATEST_DUMP out0/1/2.f32, NHWC, 255 channels).
usage: head.py <dir> [seg:file:H:W ...]   (convref.py <dir> <seg> first, for seg<N>_meta.npz)
default: 52:out0.f32:48:80 59:out1.f32:24:40 66:out2.f32:12:20  (yolov5n, 640x384)
"""
import sys

import numpy as np

REC = np.dtype([("op", "<u4"), ("imm", "<u4"), ("vec", "u1", 64)])


def main():
    d = sys.argv[1]
    heads = [a.split(":") for a in sys.argv[2:]] or \
        [("52", "out0.f32", "48", "80"), ("59", "out1.f32", "24", "40"), ("66", "out2.f32", "12", "20")]
    ops = np.fromfile(f"{d}/nn.bin", dtype=REC)
    r = list(np.load(f"{d}/resets.npy")) + [len(ops)]
    blob = open(f"{d}/yolov5n_t41.bin", "rb").read()
    for L, fn, h, w in heads:
        L = int(L)
        m = np.load(f"{d}/seg{L}_meta.npz")
        wend, noc = int(m["wend"]), int(m["noc"])
        nb = (noc + 31) // 32
        B = np.concatenate([np.frombuffer(blob[wend + 0x25 + 256 * k:wend + 0xa5 + 256 * k], "<i4")
                            for k in range(nb)]).astype(np.int64)
        Fl = np.concatenate([np.frombuffer(blob[wend + 0xa5 + 256 * k:wend + 0x125 + 256 * k], "<f4")
                             for k in range(nb)])
        out = np.fromfile(f"{d}/{fn}", "<f4").reshape(int(h) * int(w), noc)
        s = ops[r[L]:r[L + 1]]
        v = s["vec"][(s["op"] == 3) & (s["imm"] < 0x10)].copy().view("<i4").reshape(-1, 8, 32)
        ob = m["cpoc_ob"]
        val = ((v.astype(np.int64) + B.reshape(-1, 32)[ob][:, None, :]).astype(np.float32)
               * Fl.reshape(-1, 32)[ob][:, None, :]).astype(np.float32)
        # each 32-lane block of each traced pixel must be a row block of the dump (local tiling:
        # positions come from the DMA plan, so this is a set check)
        sets = {k: set(map(bytes, np.ascontiguousarray(out[:, 32 * k:min(32 * k + 32, noc)])))
                for k in range(nb)}
        hit = tot = 0
        for i in range(len(val)):
            k = int(ob[i])
            n = min(32, noc - 32 * k)
            for p in range(8):
                tot += 1
                hit += val[i, p, :n].tobytes() in sets[k]
        print(f"L{L} {m['name']} -> {fn}: float32(S + B) * Fl found {hit}/{tot}")


if __name__ == "__main__":
    main()
