#!/usr/bin/env python3
"""onna text graph for a venus .bin of the QSUM kind (NNA requantizes; ReLU; per-channel 1x1 convs,
depthwise 3x3, 2x2 maxpool, ConvTranspose 2x2/2, concat), from its ONNX.

usage: mkgraphq.py <model.onnx> > model.qgraph

Records are the W blocks in .bin file order, which is the ONNX order of Conv/ConvTranspose nodes;
concat records are the header concat records in file order. Format (tensors numbered, 0 = input):
  onnaq-graph 1
  tensors <n>
  input 0 <H> <W> 3                     # RGB bytes
  stem <out> <in> <rec> <oc>            # 3x3/2 conv, pad bottom/right 1, as a 1x1 on 32-byte pixels
  conv <out> <in> <rec> <oc> <ic> <k> <pad> <dw>    # QSUM (ReLU folded into B/T); dw: depthwise
  pc <out> <in> <rec> <c>               # per-channel 1x1 (a byte -> byte table per channel)
  head <out> <in> <rec> <oc> <ic>       # 1x1, float out = f32(S + B) * Fl
  maxpool2 <out> <in>
  deconv2 <out> <in> <rec> <c>          # ConvTranspose 2x2/2 per channel (stored flipped)
  concat <out> <n> <in>... <crec>
  output <t0> <t1>
"""
import sys

import onnx
from onnx import numpy_helper


def main():
    m = onnx.load(sys.argv[1])
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    outs = [o.name for o in m.graph.output]
    dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    tid = {m.graph.input[0].name: 0}

    def t(name):
        if name not in tid:
            tid[name] = len(tid)
        return tid[name]

    lines = [f"input 0 {dims[2]} {dims[3]} 3"]
    rec = crec = 0
    alias = {}
    heads = {}
    for n in m.graph.node:
        a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
        src = alias.get(n.input[0], n.input[0])
        if n.op_type == "Conv":
            w = init[n.input[1]]
            O, I, kh, kw = w.shape
            g = a.get("group", 1)
            if rec == 0:
                assert (kh, kw, a["strides"], a["pads"]) == (3, 3, [2, 2], [0, 0, 1, 1]), "stem shape"
                lines.append(f"stem {t(n.output[0])} {t(src)} {rec} {O}")
            elif n.output[0] in outs:
                assert g == 1 and kh == kw == 1
                heads[n.output[0]] = t(n.output[0])
                lines.append(f"head {t(n.output[0])} {t(src)} {rec} {O} {I}")
            elif g > 1 and kh == kw == 1:
                assert g == O == I * g // I and I == 1, "per-channel 1x1"
                lines.append(f"pc {t(n.output[0])} {t(src)} {rec} {O}")
            else:
                assert a.get("strides", [1, 1]) == [1, 1] and kh == kw and kh in (1, 3)
                p = a.get("pads", [0, 0, 0, 0])
                assert len(set(p)) == 1
                dw = int(g > 1)
                assert g in (1, O) and (not dw or I == 1)
                lines.append(f"conv {t(n.output[0])} {t(src)} {rec} {O} {I * g if dw else I} {kh} {p[0]} {dw}")
            rec += 1
        elif n.op_type == "ConvTranspose":
            w = init[n.input[1]]
            assert w.shape[1:] == (1, 2, 2) and a["strides"] == [2, 2] and a.get("group") == w.shape[0]
            lines.append(f"deconv2 {t(n.output[0])} {t(src)} {rec} {w.shape[0]}")
            rec += 1
        elif n.op_type == "Relu":
            alias[n.output[0]] = src          # folded into the producer's B/T
        elif n.op_type == "MaxPool":
            assert a["kernel_shape"] == [2, 2] and a["strides"] == [2, 2]
            lines.append(f"maxpool2 {t(n.output[0])} {t(src)}")
        elif n.op_type == "Concat":
            assert a["axis"] == 1
            ins = [t(alias.get(i, i)) for i in n.input]
            lines.append(f"concat {t(n.output[0])} {len(ins)} " + " ".join(map(str, ins)) + f" {crec}")
            crec += 1
        else:
            raise SystemExit(f"mkgraphq: unsupported op {n.op_type} ({n.name})")
    lines.append("output " + " ".join(str(heads[o]) for o in outs))
    print("onnaq-graph 1")
    print(f"tensors {len(tid)}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
