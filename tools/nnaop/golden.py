#!/usr/bin/env python3
"""Golden model of yolov5n on the T41 (venus + NNA arithmetic), from the .bin alone: no trace.

  golden.py <model.bin> <model.onnx> <input_rgba.raw 640x384> [out0.f32 out1.f32 out2.f32]

All parameters come from binwalk.py (the .bin); the ONNX supplies the graph. Ops as in
docs/nna-runtime.md: int conv (pad 128; the stem reads the raw image, pad 0) -> float32 requant
+ 32-segment SiLU; adds (NNA QSUM, A = the shortcut); concat rescale; 5x5 maxpool; nearest x2;
head float32(S + B) * Fl. Given the reference head outputs, it compares them BY POSITION.
"""
import sys

import numpy as np
import onnx

sys.path.insert(0, __import__("os").path.dirname(__file__) or ".")
from binwalk import build  # noqa: E402

f32 = np.float32


def conv_int(x, W, st, pd, padval=128):
    H, Wd, C = x.shape
    O, Ci, kh, kw = W.shape
    xp = np.full((H + 2 * pd + st, Wd + 2 * pd + st, C), padval, np.int64)
    xp[pd:pd + H, pd:pd + Wd] = x
    OH, OW = (H + 2 * pd - kh) // st + 1, (Wd + 2 * pd - kw) // st + 1
    S = np.zeros((OH, OW, O), np.int64)
    for ky in range(kh):
        for kx in range(kw):
            S += np.einsum("hwc,oc->hwo", xp[ky:ky + st * OH:st, kx:kx + st * OW:st, :Ci],
                           W[:, :, ky, kx])
    return S


def requant(S, B, Fl, table):
    dy, ty = table
    t = np.rint((S + B).astype(f32) * Fl).astype(np.int64)
    seg = t >> 10
    sc = np.clip(seg, 0, 31)
    y = ty[sc].astype(f32) + dy[sc].astype(f32) * (np.clip(t, 0, 32767) - sc * 1024).astype(f32) / f32(1024)
    y = np.where(seg < 0, f32(ty[0]), y)
    return np.clip(np.rint(y / f32(128)), 0, 255).astype(np.int64)


def run(blob, onnx_path, img):
    info = build(blob, onnx_path)
    cv, adds, cat = info["convs"], info["adds"], info["concat_a"]
    stem = next(L for k, L in cv.items() if k.startswith("?"))      # the 6x6 stem record
    G = stem["G"]
    W0 = np.zeros((16, 4, 6, 6), np.int64)                         # group = kx, ic = 4*ky + c
    for kx in range(6):
        for ky in range(6):
            for c in range(4):
                W0[:, c, ky, kx] = G[kx, :16, 4 * ky + c]
    model = onnx.load(onnx_path)
    mul_of = {n.input[0]: n.output[0] for n in model.graph.node
              if n.op_type == "Mul" and len(n.input) == 2 and "/act/" in n.name}
    T = {"images": img.astype(np.int64)}
    heads = {}
    for n in model.graph.node:
        name = n.name
        if n.op_type == "Conv":
            x = T[n.input[0]]
            if name == "/model.0/conv/Conv":
                q = requant(conv_int(x, W0, 2, 2, padval=0), stem["B"][:16], stem["Fl"][:16], stem["table"])
            else:
                L = cv[name]
                noc = L["noc"]
                S = conv_int(x, L["Wq"], L["stride"], L["pad"])
                if name.startswith("/model.24/"):
                    heads[name] = ((S + L["B"][:noc]).astype(f32) * L["Fl"][:noc]).astype(f32)
                    continue
                q = requant(S, L["B"][:noc], L["Fl"][:noc], L["table"])
            T[mul_of[n.output[0]]] = q
        elif n.op_type == "Add" and not name.startswith("/model.24"):
            a = adds[name]
            T[n.output[0]] = np.clip(np.floor_divide(64 * (a["wa"] * T[n.input[0]] + a["wb"] * T[n.input[1]])
                                                      + a["B"], a["T"]) + 128, 0, 255)
        elif n.op_type == "Concat" and name in cat:
            T[n.output[0]] = np.concatenate(
                [x if f == 1 else np.clip(np.rint((x.astype(f32) - f32(128)) * f) + 128, 0, 255).astype(np.int64)
                 for x, f in zip((T[i] for i in n.input), cat[name])], axis=-1)
        elif n.op_type == "MaxPool":
            x = T[n.input[0]]
            H, Wd, C = x.shape
            xp = np.full((H + 4, Wd + 4, C), -1, np.int64)
            xp[2:H + 2, 2:Wd + 2] = x
            out = np.full_like(x, -1)
            for ky in range(5):
                for kx in range(5):
                    out = np.maximum(out, xp[ky:ky + H, kx:kx + Wd])
            T[n.output[0]] = out
        elif n.op_type == "Resize":
            T[n.output[0]] = T[n.input[0]].repeat(2, axis=0).repeat(2, axis=1)
    return [heads[f"/model.24/m.{i}/Conv"] for i in range(3)]


def main():
    blob = open(sys.argv[1], "rb").read()
    img = np.fromfile(sys.argv[3], np.uint8).reshape(384, 640, 4)
    outs = run(blob, sys.argv[2], img)
    for i, (h, ref) in enumerate(zip(outs, sys.argv[4:7])):
        r = np.fromfile(ref, "<f4").reshape(h.shape)
        eq = h.view(np.uint32) == r.view(np.uint32)
        print(f"head {i} {h.shape}: bit-identical to {ref}: {int(eq.sum())}/{eq.size}"
              f" ({'EXACT' if eq.all() else 'max |diff| %.6f' % float(np.abs(h - r).max())})")


if __name__ == "__main__":
    main()
