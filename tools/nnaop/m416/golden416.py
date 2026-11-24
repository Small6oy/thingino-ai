#!/usr/bin/env python3
"""Golden model of a venus T41 .bin of the ReLU/QSUM kind (the thingino-ai 416x320 detector),
from the .bin, the ONNX graph (structure only) and an RGB frame. No trace needed.

Every conv/deconv except the heads requantizes on the NNA (QSUM through BT):
    q = clamp(floor((4 * sum(w * x) + B) / T) + 128, 0, 255)      (x raw u8, w = raw - 128)
the heads read the raw sums:  out = float32(S + B) * Fl.
2x2 maxpool = byte max, concat = byte concat. Padding is 0.

usage: golden416.py <model.bin> <model.onnx> <frame.rgb> [outprefix]
"""
import struct
import sys

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, __import__("os").path.dirname(__file__) or ".")
from walk416 import layers  # noqa: E402

IH, IW = 320, 416
VAR = {}   # test variants
# stem: input slot ic -> (c, ky, kx); None = unused (weight raw 0)
STEM = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (2, 0, 1), (1, 1, 0),
        (0, 0, 2), (1, 0, 2), (2, 0, 2), (2, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1), None,
        (0, 2, 0), (1, 2, 0), (2, 2, 0), None, (0, 2, 1), (1, 2, 1), (2, 2, 1), None,
        (0, 2, 2), (1, 2, 2), (2, 2, 2), None, (0, 1, 2), (1, 1, 2), (2, 1, 2), None]


def slices(raw, ic):
    """2-bit-slice groups -> int weights [n, 32 oc, ic]. raw [n, 4 slices, bytes]:
    ic 32: a slice is a 256-B entry [32 oc][8 B: 32 ic x 2 bit]; ic 1: 8 B [32 oc x 2 bit]"""
    n = raw.shape[0]
    v = np.zeros((n, 32, ic), np.int64)
    for k in range(4):
        e = raw[:, k].reshape(n, -1) if ic == 1 else raw[:, k].reshape(n, 32, 8)
        for oc in range(32):
            for i in range(ic):
                b = e[:, oc // 4] >> (2 * (oc % 4)) if ic == 1 else e[:, oc, i // 4] >> (2 * (i % 4))
                v[:, oc, i] |= (b & 3).astype(np.int64) << (2 * k)
    return v - 128


def load(blob):
    out = []
    for L in layers(blob):
        w = np.frombuffer(blob[L["w"]:L["w"] + L["nbytes"]], np.uint8)
        nob, nic, taps = L["nob"], L["nic"], L["kh"] * L["kw"]
        if L["dw"]:
            G = slices(w.reshape(-1, 4, 8), 1)[:, :, 0].reshape(nob, taps, 32)   # [block, tap, oc]
            W = G.transpose(0, 2, 1).reshape(nob * 32, L["kh"], L["kw"])         # [c, ky, kx]
        else:
            G = slices(w.reshape(-1, 4, 256), 32).reshape(nob, nic, taps, 32, 32)  # [ob, ic chunk, tap, oc, ic]
            W = G.transpose(0, 3, 1, 4, 2).reshape(nob * 32, nic * 32, L["kh"], L["kw"])
        p = L["p"]
        B = np.concatenate([np.frombuffer(blob[p + 0x25 + 256 * k:p + 0xa5 + 256 * k], "<i4") for k in range(nob)])
        T = np.concatenate([np.frombuffer(blob[p + 0xa5 + 256 * k:p + 0x125 + 256 * k], "<i4") for k in range(nob)])
        Fl = np.concatenate([np.frombuffer(blob[p + 0xa5 + 256 * k:p + 0x125 + 256 * k], "<f4") for k in range(nob)])
        out.append(dict(L=L, W=W, B=B.astype(np.int64), T=(T & 0x3ffffff).astype(np.int64), Fl=Fl))
    return out


def concat_records(blob):
    """header concat records [n, 0 x n, n] + float32 s_1..s_n, 1/s_out (zero point 0), file order"""
    out = []
    for nin in range(2, 6):
        pat = struct.pack("<i", nin) + struct.pack("<i", 0) * nin + struct.pack("<i", nin)
        o = blob.find(pat)
        while o >= 0:
            f = struct.unpack("<" + "f" * (nin + 1), blob[o + len(pat):o + len(pat) + 4 * (nin + 1)])
            if all(1e-6 < x < 1e3 for x in f):
                out.append((o, [np.float32(np.float32(f[i]) * np.float32(f[nin])) for i in range(nin)]))
            o = blob.find(pat, o + 1)
    return [a for _, a in sorted(out)]


def qsum(S, B, T):
    r = 4 * S + B
    q = np.floor_divide(r, np.where(T == 0, 1, T))
    q = np.where(T == 0, np.where(r >= 0, 127, -128), q)
    return np.clip(q + 128, 0, 255).astype(np.uint8)


def conv(x, W, pad):
    """x [H, W, C] u8, W [O, C, kh, kw] (dense) -> int64 sums [H', W', O], stride 1"""
    H, Wd, C = x.shape
    O, _, kh, kw = W.shape
    xp = np.zeros((H + 2 * pad, Wd + 2 * pad, C), np.int64)
    xp[pad:pad + H, pad:pad + Wd] = x
    S = np.zeros((H + 2 * pad - kh + 1, Wd + 2 * pad - kw + 1, O), np.int64)
    for ky in range(kh):
        for kx in range(kw):
            S += xp[ky:ky + S.shape[0], kx:kx + S.shape[1]] @ W[:, :C, ky, kx].T
    return S


def dwconv(x, W, pad):
    H, Wd, C = x.shape
    _, kh, kw = W.shape
    xp = np.zeros((H + 2 * pad, Wd + 2 * pad, C), np.int64)
    xp[pad:pad + H, pad:pad + Wd] = x
    S = np.zeros((H + 2 * pad - kh + 1, Wd + 2 * pad - kw + 1, C), np.int64)
    for ky in range(kh):
        for kx in range(kw):
            S += xp[ky:ky + S.shape[0], kx:kx + S.shape[1]] * W[:C, ky, kx]
    return S


def stem_f(rgb):
    """[160, 208, 32] u8: the stem's 32-byte pixels (3x3 s2, pad bottom/right 1)"""
    xp = np.zeros((IH + 1, IW + 1, 3), np.uint8)
    xp[:IH, :IW] = rgb
    f = np.zeros((IH // 2, IW // 2, 32), np.uint8)
    for i, s in enumerate(STEM):
        if s is None:
            continue
        c, ky, kx = s
        f[:, :, i] = xp[ky:ky + IH:2, kx:kx + IW:2, c]
    return f


def run(blob, onnx_path, rgb, trace_cb=None):
    P = load(blob)
    m = onnx.load(onnx_path)
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    t = {}
    k = 0
    cats = concat_records(blob)
    heads = {}
    for n in m.graph.node:
        a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
        if n.op_type in ("Conv", "ConvTranspose"):
            L = P[k]
            O = init[n.input[1]].shape[0]
            if k == 0:
                S = np.einsum("hwi,oi->hwo", stem_f(rgb).astype(np.int64), L["W"][:, :, 0, 0])
            elif n.op_type == "ConvTranspose":
                x = t[n.input[0]]
                H, Wd, C = x.shape
                S = np.zeros((2 * H, 2 * Wd, C), np.int64)
                for dy in range(2):
                    for dx in range(2):
                        ph = VAR.get("phase", (0, 0))
                        fy, fx = (1 - dy) ^ ph[0], (1 - dx) ^ ph[1]
                        S[dy::2, dx::2] = x.astype(np.int64) * L["W"][:C, fy, fx]   # stored flipped (conv over the dilated input)
                        if VAR.get("nearest"):
                            S[dy::2, dx::2] = x.astype(np.int64) * L["W"][:C, 1, 1]
            elif a.get("group", 1) > 1:
                S = dwconv(t[n.input[0]], L["W"], a["pads"][0])
            else:
                S = conv(t[n.input[0]], L["W"], a["pads"][0])
            S = S[:, :, :O]
            if n.output[0] in [o.name for o in m.graph.output]:
                heads[n.output[0]] = ((S + L["B"][:O]).astype(np.float32) * L["Fl"][:O]).astype(np.float32)
            else:
                t[n.output[0]] = qsum(S, L["B"][:O], L["T"][:O])
                if trace_cb:
                    trace_cb(k, t[n.output[0]])
            k += 1
        elif n.op_type == "Relu":
            t[n.output[0]] = t[n.input[0]]          # folded into BT
        elif n.op_type == "MaxPool":
            x = t[n.input[0]]
            H, Wd = x.shape[0] // 2 * 2, x.shape[1] // 2 * 2
            x = x[:H, :Wd]
            t[n.output[0]] = np.maximum(np.maximum(x[0::2, 0::2], x[0::2, 1::2]),
                                        np.maximum(x[1::2, 0::2], x[1::2, 1::2]))
        elif n.op_type == "Concat":
            # header concat record: n, zp x n, n, s_1..s_n, 1/s_out; input i -> rint(x * s_i / s_out)
            a = cats.pop(0)
            assert len(a) == len(n.input), "concat record arity"
            t[n.output[0]] = np.concatenate(
                [np.clip(np.rint(t[i].astype(np.float32) * ai), 0, 255).astype(np.uint8)
                 for i, ai in zip(n.input, a)], axis=2)
        else:
            raise SystemExit(f"unknown op {n.op_type}")
    if VAR.get("keep") is not None:
        VAR["keep"].update(t)
    return heads


def main():
    blob = open(sys.argv[1], "rb").read()
    rgb = np.fromfile(sys.argv[3], np.uint8).reshape(IH, IW, 3)
    heads = run(blob, sys.argv[2], rgb)
    pre = sys.argv[4] if len(sys.argv) > 4 else None
    for name, h in heads.items():
        print(name, h.shape, f"{h.min():.3f}..{h.max():.3f}")
        if pre:
            h.tofile(f"{pre}.{name}.f32")


if __name__ == "__main__":
    main()
