#!/usr/bin/env python3
"""Turn a YOLOv5 ONNX + its venus .bin into onna's text graph (the network structure plus which
.bin record each op uses). The numbers themselves stay in the .bin; onna's C loader reads them.

usage: mkgraph.py <model.bin> <model.onnx> > model.graph

Format (one op per line, tensors numbered; tensor 0 = the AIP input, H W C on the input line):
  onna-graph 1
  input 0 <H> <W> <C>
  conv <out> <in> <rec> <oc> <ic> <kh> <kw> <stride> <pad> <padval> <stem 0|1> <head 0|1> <nchunks> <c0>:<k0> ...
  add <out> <a> <b> <rec>                  # a = the shortcut (QSUM weight wA)
  concat <out> <n> <in>... <src>...        # src per input: 1 | h<k>.<i> | t<convrec>
  maxpool5 <out> <in>
  upsample2 <out> <in>
  output <t0> <t1> <t2>
<rec>: conv records in .bin file order; add records in header order; h<k>: the k-th 2-input
header concat record; t<r>: the type-30 record after conv record r (its factor array).
"""
import sys

import onnx

sys.path.insert(0, __import__("os").path.dirname(__file__) or ".")
from binwalk import build, conv_layers  # noqa: E402


def main():
    blob = open(sys.argv[1], "rb").read()
    info = build(blob, sys.argv[2])
    recs = conv_layers(blob)
    rec_of_w = {L["wstart"]: i for i, L in enumerate(recs)}
    model = onnx.load(sys.argv[2])
    tid = {"images": 0}

    def t(name):
        if name not in tid:
            tid[name] = len(tid)
        return tid[name]

    mul_of = {n.input[0]: n.output[0] for n in model.graph.node
              if n.op_type == "Mul" and len(n.input) == 2 and "/act/" in n.name}
    add_names = list(info["adds"].keys())
    # the concat factor sources, as binwalk derived them
    hdr_order = [k for k, v in info["concat_a"].items()]
    lines = ["onna-graph 1", "input 0 384 640 4"]
    heads = []
    nh = 0
    prod = {o: n for n in model.graph.node for o in n.output}
    for n in model.graph.node:
        nm = n.name
        if n.op_type == "Conv":
            if nm == "/model.0/conv/Conv":
                L = next(v for k, v in info["convs"].items() if k.startswith("?"))
                rec = rec_of_w[L["wstart"]]
                out = t(mul_of[n.output[0]])
                lines.append(f"conv {out} {t(n.input[0])} {rec} 16 4 6 6 2 2 0 1 0 1 0:4")
                continue
            L = info["convs"][nm]
            rec = rec_of_w[L["wstart"]]
            O, C, kh, kw = L["Wq"].shape
            head = nm.startswith("/model.24/")
            out = t(n.output[0]) if head else t(mul_of[n.output[0]])
            if head:
                heads.append(out)
            # input chunks exactly as binwalk assembled the weights
            src = prod.get(n.input[0])
            ch = []
            if src is not None and src.op_type == "Concat":
                base = 0
                for i in src.input:
                    ci = _channels(model, i)
                    ch += [(base + c, min(32, ci - c)) for c in range(0, ci, 32)]
                    base += ci
            else:
                ch = [(c, min(32, C - c)) for c in range(0, C, 32)]
            lines.append(f"conv {out} {t(n.input[0])} {rec} {O} {C} {kh} {kw} {L['stride']} {L['pad']}"
                         f" 128 0 {int(head)} {len(ch)} " + " ".join(f"{c}:{k}" for c, k in ch))
        elif n.op_type == "Add" and not nm.startswith("/model.24"):
            lines.append(f"add {t(n.output[0])} {t(n.input[0])} {t(n.input[1])} {add_names.index(nm)}")
        elif n.op_type == "Concat" and nm in info["concat_a"]:
            fs = info["concat_a"][nm]
            srcs = []
            for i, f in enumerate(fs):
                if f == 1:
                    srcs.append("1")
                elif nm in info.get("concat_src", {}):
                    srcs.append(info["concat_src"][nm][i])
                else:
                    srcs.append("?")
            lines.append(f"concat {t(n.output[0])} {len(n.input)} " + " ".join(str(t(i)) for i in n.input)
                         + " " + " ".join(srcs))
        elif n.op_type == "MaxPool":
            lines.append(f"maxpool5 {t(n.output[0])} {t(n.input[0])}")
        elif n.op_type == "Resize":
            lines.append(f"upsample2 {t(n.output[0])} {t(n.input[0])}")
    lines.append("output " + " ".join(str(h) for h in heads))
    lines.insert(1, f"tensors {len(tid)}")
    print("\n".join(lines))
    del hdr_order, nh


_CH = {}


def _channels(model, name):
    """channel count of a tensor (by walking the graph once)"""
    if not _CH:
        from onnx import numpy_helper
        init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        _CH["images"] = 3
        for n in model.graph.node:
            if n.op_type == "Conv":
                _CH[n.output[0]] = init[n.input[1]].shape[0]
            elif n.op_type in ("Sigmoid", "MaxPool", "Resize", "Mul", "Add") and n.input[0] in _CH:
                _CH[n.output[0]] = _CH[n.input[0]]
            elif n.op_type == "Concat" and all(i in _CH for i in n.input):
                _CH[n.output[0]] = sum(_CH[i] for i in n.input)
    return _CH[name]


if __name__ == "__main__":
    main()
