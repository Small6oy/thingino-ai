#!/usr/bin/env python3
"""Read a venus (magik-transform-tools 2.0.4) T41 .bin without a trace: every conv's weights,
requant parameters and SiLU table, the residual-add records and the concat rescale factors.

Layout found by tracing yolov5n (docs/nna-runtime.md):
  W block:      preceded by int32 [bytes, 7, nob, 1, kh, kw, 4, 32, 32], then bytes/256 entries
  param record: at the W block's end (wend): 02 00 00 00 02 00 00 00 01 00 ..., count 32 at
                +0x21, then per 32-oc block k: int32 B[32] at +0x25+256k, float32 Fl[32] at
                +0xa5+256k, then the SiLU table after 04 00 00 00 20 00 00 00
  W groups:     oc block -> input chunk (32 channels; concat members chunked separately) -> taps
  add records:  2, wA, wB (int32), 2, sA, sB, B, T (float32), in network order
  concat:       header records n, -128 x n, n, s_1..s_n, 1/s_out; or, for a 2-input concat whose
                input 0 passes through, a per-channel factor in the type-30 record after the
                tables of input 0's conv (applied to input 1)
The layers are matched to the ONNX graph (needed anyway for the network structure): convs by
shape and weights (+-1), adds and concats by order.     usage: binwalk.py <model.bin> <model.onnx>
"""
import struct
import sys

import numpy as np
import onnx
from onnx import numpy_helper

SIG = b"\x02\x00\x00\x00\x02\x00\x00\x00\x01\x00"
MARK = b"\x04\x00\x00\x00\x20\x00\x00\x00"


def i32(b, o):
    return struct.unpack("<i", b[o:o + 4])[0]


def f32(b, o):
    return struct.unpack("<f", b[o:o + 4])[0]


def table32(blob, o):
    t = np.frombuffer(blob[o:o + 128], "<i2").astype(np.float64)
    return np.concatenate([t[0:32:2], t[1:32:2]]), np.concatenate([t[32::2], t[33::2]])


def wgroups(raw, wbit=8):
    sl = wbit // 2
    G = np.zeros((len(raw) // sl, 32, 32), np.int64)
    for g in range(len(G)):
        for k in range(sl):
            e = raw[g * sl + k].reshape(32, 8).astype(np.int64)
            for ic in range(32):
                G[g, :, ic] |= ((e[:, ic // 4] >> (2 * (ic % 4))) & 3) << (2 * k)
    return G - (1 << (wbit - 1))


def conv_layers(blob):
    """every conv in file order: W block, shape fields, params, table"""
    out = []
    o = blob.find(SIG)
    while o >= 0:
        if i32(blob, o + 0x21) == 32:
            wend = o
            for n in range(1, 8192):
                ws = wend - 256 * n
                if ws < 36:
                    break
                h = [i32(blob, ws - 36 + 4 * k) for k in range(9)]
                if h[0] == 256 * n and h[1] == 7 and h[6] == 4 and h[7] == 32 and h[8] == 32:
                    nob, kh, kw = h[2], h[4], h[5]
                    raw = np.frombuffer(blob[ws:wend], np.uint8).reshape(n, 256)
                    B = np.concatenate([np.frombuffer(blob[wend + 0x25 + 256 * k:wend + 0xa5 + 256 * k], "<i4")
                                        for k in range(nob)]).astype(np.int64)
                    Fl = np.concatenate([np.frombuffer(blob[wend + 0xa5 + 256 * k:wend + 0x125 + 256 * k], "<f4")
                                         for k in range(nob)])
                    mk = blob.find(MARK, wend + 0x25 + 256 * nob - 0x80)
                    out.append(dict(wstart=ws, wend=wend, n=n, nob=nob, kh=kh, kw=kw,
                                    G=wgroups(raw), B=B, Fl=Fl, table=table32(blob, mk + 8),
                                    after=mk + 8 + 256))
                    break
        o = blob.find(SIG, o + 1)
    return out


def onnx_info(path):
    m = onnx.load(path)
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    prod = {o: n for n in m.graph.node for o in n.output}
    convs = [n for n in m.graph.node if n.op_type == "Conv"]
    return m, init, prod, convs


def build(blob, onnx_path):
    m, init, prod, convs = onnx_info(onnx_path)
    # channel count of every tensor (forward over the graph by shapes of weights)
    ch = {"images": 3}
    for n in m.graph.node:
        if n.op_type == "Conv":
            ch[n.output[0]] = init[n.input[1]].shape[0]
        elif n.op_type in ("Sigmoid", "MaxPool", "Resize") and n.input[0] in ch:
            ch[n.output[0]] = ch[n.input[0]]
        elif n.op_type == "Mul" and n.input[0] in ch:
            ch[n.output[0]] = ch[n.input[0]]
        elif n.op_type == "Add" and n.input[0] in ch:
            ch[n.output[0]] = ch[n.input[0]]
        elif n.op_type == "Concat" and all(i in ch for i in n.input):
            ch[n.output[0]] = sum(ch[i] for i in n.input)

    def chunks(t):
        n = prod.get(t)
        if n is not None and n.op_type == "Concat":
            out, base = [], 0
            for i in n.input:
                out += [(base + c, min(32, ch[i] - c)) for c in range(0, ch[i], 32)]
                base += ch[i]
            return out
        return [(c, min(32, ch[t] - c)) for c in range(0, ch[t], 32)]

    layers = conv_layers(blob)
    res = {}
    used = set()
    for li, L in enumerate(layers):
        best = None
        for n in convs:
            if n.name in used or n.name == "/model.0/conv/Conv":
                continue
            w = init[n.input[1]].astype(np.float64)
            O, C, kh, kw = w.shape
            if (kh, kw) != (L["kh"], L["kw"]) or (O + 31) // 32 != L["nob"]:
                continue
            ck = chunks(n.input[0])
            if len(L["G"]) != L["nob"] * len(ck) * kh * kw:
                continue
            s = np.abs(w).reshape(O, -1).max(1) / 127
            q = np.round(w / s[:, None, None, None]).astype(np.int64)
            Wq = np.zeros(q.shape, np.int64)
            g = 0
            for ob in range(L["nob"]):
                nr = min(32, O - 32 * ob)
                for c0, k in ck:
                    for ky in range(kh):
                        for kx in range(kw):
                            Wq[32 * ob:32 * ob + nr, c0:c0 + k, ky, kx] = L["G"][g, :nr, :k]
                            g += 1
            err = int(np.abs(Wq - q).max())
            if best is None or err < best[0]:
                best = (err, n, Wq)
        if best is None:
            res[f"?{li}"] = L
            continue
        err, n, Wq = best
        used.add(n.name)
        L.update(name=n.name, Wq=Wq, err=err, noc=Wq.shape[0],
                 stride=[a for a in n.attribute if a.name == "strides"][0].ints[0],
                 pad=[a for a in n.attribute if a.name == "pads"][0].ints[0])
        res[n.name] = L

    if "-v" in sys.argv:
        for k, L in res.items():
            print(f"  W@{L['wstart']:8} n {L['n']:4} nob {L['nob']} k {L['kh']}x{L['kw']} -> {k}"
                  f" (err {L.get('err', '-')})")
        print("  ONNX convs not matched:", [n.name for n in convs if n.name not in used])
    # residual adds, in order
    adds = []
    for o in range(0, min(len(blob), 65536) - 40):
        c1, wa, wb, c2 = struct.unpack("<iiii", blob[o:o + 16])
        if c1 == 2 and c2 == 2 and 0 < wa <= 255 and 0 < wb <= 255 and max(wa, wb) == 255:
            sa, sb, B, T = struct.unpack("<ffff", blob[o + 16:o + 32])
            if 1e-4 < sa < 1 and 1e-4 < sb < 1 and B < 0 and 1000 < T < 100000:
                adds.append(dict(off=o, wa=wa, wb=wb, B=int(B), T=int(T)))
    add_nodes = [n.name for n in m.graph.node if n.op_type == "Add" and not n.name.startswith("/model.24")]

    # concats: header records (n >= 2 inputs) in order
    hdr = []
    for nin in range(2, 6):
        pat = struct.pack("<i", nin) + struct.pack("<i", -128) * nin + struct.pack("<i", nin)
        o = blob.find(pat)
        while o >= 0:
            f = struct.unpack("<" + "f" * (nin + 1), blob[o + len(pat):o + len(pat) + 4 * (nin + 1)])
            hdr.append((o, [np.float32(f[i]) * np.float32(f[nin]) for i in range(nin)]))
            o = blob.find(pat, o + 1)
    hdr.sort()
    concat_a = {}
    cross = []
    for n in m.graph.node:
        if n.op_type != "Concat" or n.name.startswith("/model.24") or \
                onnx.helper.get_attribute_value(n.attribute[0]) != 1:      # channel concats only
            continue
        blk = n.name.split("/")[1]
        same = all(i.split("/")[1] == blk for i in n.input)
        if same:
            concat_a[n.name] = [np.float32(1)] * len(n.input)
        else:
            cross.append(n)
    hdr2 = [a for _, a in hdr if len(a) == 2]
    for n in cross:
        if hdr2:
            concat_a[n.name] = hdr2.pop(0)
            continue
        # type-30 record after input 0's conv: the factor for input 1
        p0 = prod[n.input[0]]
        conv0 = prod[p0.input[0]] if p0.op_type == "Mul" else p0
        L = res[conv0.name]
        a0 = L["after"]
        assert i32(blob, a0) == 30, f"{n.name}: no type-30 record after {conv0.name}"
        fac = None
        for off in range(4):
            h = np.frombuffer(blob[a0 + off:a0 + off + 4000], "<f4")
            vals, cnt = np.unique(h[np.isfinite(h)], return_counts=True)
            for v, c in zip(vals, cnt):
                if c >= 32 and 0.3 < v < 1.5:
                    fac = np.float32(v)
        concat_a[n.name] = [np.float32(1), fac]
    return dict(convs=res, adds=dict(zip(add_nodes, adds)), concat_a=concat_a)


def main():
    blob = open(sys.argv[1], "rb").read()
    info = build(blob, sys.argv[2])
    cv = info["convs"]
    unmatched = [k for k in cv if k.startswith("?")]
    print(f"{len(cv)} conv records in the .bin; matched to ONNX: {len(cv) - len(unmatched)};"
          f" max weight error vs ONNX: {max(L['err'] for k, L in cv.items() if not k.startswith('?'))}")
    print(f"adds: {len(info['adds'])}; concat factors:",
          {k: [round(float(x), 7) for x in v] for k, v in info["concat_a"].items() if any(x != 1 for x in v)})


if __name__ == "__main__":
    main()
