#!/usr/bin/env python3
"""onna text graph for a venus .bin of the QSUM kind (NNA requantizes; ReLU; per-channel 1x1 convs,
depthwise 3x3, 2x2 maxpool, ConvTranspose 2x2/2, concat), from its ONNX.

usage: mkgraphq.py <model.onnx> [--bin model.bin] [--anchors STRIDE:w0,h0,w1,h1,w2,h2 ...] > model.qgraph

Records are the .bin's W blocks, add records and concat records, each in file order. The converter
writes them in its own execution order, which follows the ONNX order only for a straight-line graph
(the 416 detector). With --bin, each Conv is matched to its W block by its weights (same shape, best
correlation with the .bin's int8 weights); a per-channel 1x1 sits right after its producer's block;
ConvTransposes keep their relative order; adds and concats are ordered by the last block they
depend on. Without --bin: ONNX order (the straight-line case). Format (tensors numbered, 0 = input):
  onnaq-graph 1
  tensors <n>
  input 0 <H> <W> 3                     # RGB bytes
  stem <out> <in> <rec> <oc>            # 3x3/2 conv, pad bottom/right 1, as a 1x1 on 32-byte pixels
  conv <out> <in> <rec> <oc> <ic> <k> <pad> <dw> [<stride> <padb>]   # QSUM (ReLU folded into B/T);
                                        # dw: depthwise; stride 2 pads <pad> top/left, <padb> bottom/right
  pc <out> <in> <rec> <c>               # per-channel 1x1 (a byte -> byte table per channel)
  head <out> <in> <rec> <oc> <ic>       # 1x1, float out = f32(S + B) * Fl
  maxpool2 <out> <in>
  deconv2 <out> <in> <rec> <c>          # ConvTranspose 2x2/2 per channel (stored flipped)
  concat <out> <n> <in>... <crec>
  add <out> <a> <b> <arec>              # venus add record arec (ONNX order): out = clamp(floor((64 (wA a +
                                        # wB b) + B) / T) + 128); onnaq folds the export's clamp
                                        # (x + -relu(x - t)) into the conv before it
  output <t0> <t1> ...
  anchors <stride> <w0> <h0> <w1> <h1> <w2> <h2>   # optional, per head stride (input pixels);
                                        # raptor-hal's onna backend uses YOLOv5's when there are none
"""
import sys

import onnx
from onnx import numpy_helper


def bin_blocks(path):
    """the .bin's W blocks in file order: (dw, kh, nob, nic, int8 weights) as onnaq decodes them"""
    import struct
    import numpy as np
    b = open(path, "rb").read()
    out, o = [], 36
    while o + 36 <= len(b):
        nb, seven, nob, nic, kh, kw, four, tt, icg = struct.unpack_from("<9i", b, o - 36)
        taps = kh * kw
        if (seven == 7 and four == 4 and tt == 32 and icg in (1, 32) and 0 < nb <= len(b) - o
                and nb == (32 * nob * taps if icg == 1 else 1024 * nob * nic * taps)):
            raw = np.frombuffer(b, np.uint8, nb, o)
            sh = np.array([0, 2, 4, 6])
            if icg == 1:      # block -> tap -> 4 slices x 8 B (32 oc)
                e = raw.reshape(nob, taps, 4, 8)
                v = sum(((e[:, :, k, :, None] >> sh) & 3).astype(np.int32) << (2 * k) for k in range(4))
                wq = (v.reshape(nob, taps, 32) - 128).transpose(1, 0, 2).reshape(taps, 32 * nob)   # [t][oc]
            else:             # block -> input chunk -> 4 entries [32 oc][8 B]
                e = raw.reshape(nob, nic, 4, 32, 8)
                v = sum(((e[:, :, k, :, :, None] >> sh) & 3).astype(np.int32) << (2 * k) for k in range(4))
                wq = (v.reshape(nob, nic, 32, 32) - 128).transpose(0, 2, 1, 3).reshape(32 * nob, 32 * nic)  # [oc][ic]
            out.append((icg == 1, kh, nob, nic, wq))
            o += nb
            continue
        o += 1
    return out


def record_order(m, init, path):
    """{Conv/ConvTranspose name: W block}, {Add name: add record}, {Concat name: concat record}"""
    import numpy as np
    blocks = bin_blocks(path)
    alias, prod = {}, {}
    for n in m.graph.node:
        if n.op_type in ("Relu", "MaxPool"):
            alias[n.output[0]] = alias.get(n.input[0], n.input[0])
        for o in n.output:
            prod[o] = n
    convs = [n for n in m.graph.node if n.op_type in ("Conv", "ConvTranspose")]
    if len(convs) != len(blocks):
        raise SystemExit(f"mkgraphq: {len(convs)} convs, the .bin has {len(blocks)} W blocks")

    def info(n):
        a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
        w = init[n.input[1]]
        g = a.get("group", 1)
        return a, w, g

    def is_pc(n):
        a, w, g = info(n)
        return n.op_type == "Conv" and g > 1 and w.shape[2:] == (1, 1)

    rmap, used = {}, set()
    rmap[convs[0].name] = 0; used.add(0)                       # the stem: always block 0
    scores = []
    for n in convs[1:]:
        if n.op_type != "Conv" or is_pc(n):
            continue
        a, w, g = info(n)
        O, I, kh = w.shape[0], w.shape[1], w.shape[2]
        dw = g > 1
        want = ((O + 31) // 32, 1 if dw else (I + 31) // 32)
        ref = w.reshape(O, -1) if not dw else w.reshape(O, kh * kh)
        ref = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12)
        for k, (bdw, bkh, nob, nic, wq) in enumerate(blocks):
            if k in used or bdw != dw or bkh != kh or (nob, nic if not dw else 1) != want:
                continue
            q = (wq[:, :O].T if dw else wq[:O, :I]).astype(np.float64)
            q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
            scores.append((float((q * ref).sum(1).mean()), n.name, k))
    for sc, name, k in sorted(scores, reverse=True):
        if name not in rmap and k not in used:
            rmap[name] = k; used.add(k)
    def after(tn):
        """the last W block tensor tn depends on (through Relu / MaxPool / Add / Concat)"""
        tn = alias.get(tn, tn)
        n = prod.get(tn)
        if n is None:
            return -1
        if n.name in rmap:
            return rmap[n.name]
        return max(after(i) for i in n.input if i in prod)

    for n in convs:                  # per-channel 1x1: the first free block after what it depends on
        if not is_pc(n):
            continue
        p = after(n.input[0])
        k = p + 1 if p is not None else None
        while k is not None and k < len(blocks) and k in used:
            k += 1
        if k is None or k >= len(blocks) or not (blocks[k][0] and blocks[k][1] == 1):
            raise SystemExit(f"mkgraphq: no W block for per-channel {n.name}")
        rmap[n.name] = k; used.add(k)
    free = [k for k, b in enumerate(blocks) if k not in used and b[0] and b[1] == 2]
    for n in convs:                                            # ConvTranspose: their relative order
        if n.op_type == "ConvTranspose":
            rmap[n.name] = free.pop(0); used.add(rmap[n.name])
    missing = [n.name for n in convs if n.name not in rmap]
    if missing or len(used) != len(blocks):
        raise SystemExit(f"mkgraphq: unmatched convs {missing[:5]}")
    pos = {}                                                   # converter position of each tensor

    def where(tn):
        tn = alias.get(tn, tn)
        if tn not in pos:
            n = prod.get(tn)
            if n is None:
                pos[tn] = -1
            elif n.name in rmap:
                pos[tn] = rmap[n.name]
            else:
                pos[tn] = max(where(i) for i in n.input if i in prod or i == m.graph.input[0].name)
        return pos[tn]
    order = lambda op: [n.name for n in sorted((n for n in m.graph.node if n.op_type == op),
                                                key=lambda n: (max(where(i) for i in n.input), list(m.graph.node).index(n)))]
    return rmap, {nm: i for i, nm in enumerate(order("Add"))}, {nm: i for i, nm in enumerate(order("Concat"))}


def main():
    args = sys.argv[1:]
    anchors = []
    while "--anchors" in args:
        i = args.index("--anchors")
        stride, wh = args[i + 1].split(":")
        v = [float(x) for x in wh.split(",")]
        assert len(v) == 6, "--anchors STRIDE:w0,h0,w1,h1,w2,h2"
        anchors.append(f"anchors {int(stride)} " + " ".join(f"{x:g}" for x in v))
        del args[i:i + 2]
    binp = None
    if "--bin" in args:
        i = args.index("--bin")
        binp = args[i + 1]
        del args[i:i + 2]
    m = onnx.load(args[0])
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    rmap, amap, cmap = record_order(m, init, binp) if binp else (None, None, None)
    outs = [o.name for o in m.graph.output]
    dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    tid = {m.graph.input[0].name: 0}

    def t(name):
        if name not in tid:
            tid[name] = len(tid)
        return tid[name]

    lines = [f"input 0 {dims[2]} {dims[3]} 3"]
    rec = crec = arec = 0
    alias = {}
    heads = {}
    for n in m.graph.node:
        a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
        src = alias.get(n.input[0], n.input[0])
        ri = rmap[n.name] if rmap and n.op_type in ("Conv", "ConvTranspose") else rec
        if n.op_type == "Conv":
            w = init[n.input[1]]
            O, I, kh, kw = w.shape
            g = a.get("group", 1)
            if rec == 0:
                assert (kh, kw, a["strides"], a["pads"]) == (3, 3, [2, 2], [0, 0, 1, 1]), "stem shape"
                lines.append(f"stem {t(n.output[0])} {t(src)} {ri} {O}")
            elif n.output[0] in outs:
                assert g == 1 and kh == kw == 1
                heads[n.output[0]] = t(n.output[0])
                lines.append(f"head {t(n.output[0])} {t(src)} {ri} {O} {I}")
            elif g > 1 and kh == kw == 1:
                assert g == O == I * g // I and I == 1, "per-channel 1x1"
                lines.append(f"pc {t(n.output[0])} {t(src)} {ri} {O}")
            else:
                st = a.get("strides", [1, 1])
                assert st in ([1, 1], [2, 2]) and kh == kw and kh in (1, 3)
                p = a.get("pads", [0, 0, 0, 0])            # top, left, bottom, right
                dw = int(g > 1)
                assert g in (1, O) and (not dw or I == 1)
                line = f"conv {t(n.output[0])} {t(src)} {ri} {O} {I * g if dw else I} {kh} {p[0]} {dw}"
                if st == [1, 1] and len(set(p)) == 1:
                    lines.append(line)
                else:                                     # e.g. stride-2 depthwise, pads 0 0 1 1
                    assert p[0] == p[1] and p[2] == p[3], "pads must match per side pair"
                    lines.append(line + f" {st[0]} {p[2]}")
            rec += 1
        elif n.op_type == "ConvTranspose":
            w = init[n.input[1]]
            assert w.shape[1:] == (1, 2, 2) and a["strides"] == [2, 2] and a.get("group") == w.shape[0]
            lines.append(f"deconv2 {t(n.output[0])} {t(src)} {ri} {w.shape[0]}")
            rec += 1
        elif n.op_type == "Relu":
            alias[n.output[0]] = src          # folded into the producer's B/T
        elif n.op_type == "MaxPool":
            assert a["kernel_shape"] == [2, 2] and a["strides"] == [2, 2]
            lines.append(f"maxpool2 {t(n.output[0])} {t(src)}")
        elif n.op_type == "Concat":
            assert a["axis"] == 1
            ins = [t(alias.get(i, i)) for i in n.input]
            lines.append(f"concat {t(n.output[0])} {len(ins)} " + " ".join(map(str, ins)) + f" {cmap[n.name] if cmap else crec}")
            crec += 1
        elif n.op_type == "Add":
            a_, b_ = (t(alias.get(i, i)) for i in n.input)
            lines.append(f"add {t(n.output[0])} {a_} {b_} {amap[n.name] if amap else arec}")
            arec += 1
        else:
            raise SystemExit(f"mkgraphq: unsupported op {n.op_type} ({n.name})")
    lines.append("output " + " ".join(str(heads[o]) for o in outs))
    print("onnaq-graph 1")
    print(f"tensors {len(tid)}")
    print("\n".join(lines + anchors))


if __name__ == "__main__":
    main()
