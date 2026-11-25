#!/usr/bin/env python3
"""Bit-exact NumPy reference of libjzdl 5.4.0 (T31) inference for version-5 models.

  jzdl_ref.py MODEL.bin INPUT.rgb OUT.f32     # same input/output format as jzdlrun

Verified 2026-11-23 against libjzdl running under QEMU (fork, -cpu XBurstR2):
magik_model_persondet.bin, every blob and the packed output, 100 % equal on two
frames. See ../../jzdl-format.md "Arithmetic".

Activations are int8 holding u - 2^(bits-1), u = unsigned bits-wide ReLU output.
Requant (per output channel):  u = clamp(rsr(((acc + bias) * mult) >> Q, shift), 0, 2^bits - 1)
  Q = 31 for the first conv (int32 bias/mult/shift), 15 for the others (int16),
  first conv (type 46): acc wraps to int16 before the int32 bias is added; others: acc + bias wraps,
  rsr = rounding right shift (add 2^(shift-1), then >>).
Weights: 16 output channels x 2 consecutive inputs per 32-byte block.
"""
import struct, sys
import numpy as np
import jzdl


def rsr(v, s):
    return np.where(s > 0, (v + (1 << (np.maximum(s, 1) - 1))) >> s, v)


def _i(b, dt):
    return np.frombuffer(b, dt).astype(np.int64)


def requant(acc, L, q, bits):
    dt = np.int32 if q == 31 else np.int16
    bias, mult, sh = _i(L.blobs[1], dt), _i(L.blobs[2], dt), _i(L.blobs[3], dt)
    a = acc + bias
    if q == 15:   # 1x1/depthwise: the runtime wraps acc + bias to int16 [fmt: Arithmetic]
        a = ((a + 32768) & 0xffff) - 32768
        sh = sh & 15  # and takes the int16 shift modulo 16 (verified 2026-11-25: 16 acts as 0)
    return np.clip(rsr((a * mult) >> q, sh), 0, (1 << bits) - 1) - (1 << (bits - 1))


VERSION = [5]   # set by forward(): the model's version


def blocked(L):
    """weights in the v5 block layout: version 5 with the trailing flag set (persondet).
    Version 5 with flag 0 (the Magik 1.0 kit's Txx YOLOv5) stores them plain, as v4."""
    return VERSION[0] >= 5 and L.params[-1] != 0


def _pair_byte(s, nslots, o):
    """byte of slot s for output o in a 16-output group: pairs of slots in 32-byte
    blocks (o*2 + s%2); v4 hypothesis: an odd final slot sits alone in a 16-byte tail"""
    if nslots % 2 and s == nslots - 1:
        return (s // 2) * 32 + o
    return (s // 2) * 32 + o * 2 + s % 2


def first_weights(L, cin):
    """type 46. v5: tap s = ky*10 + kx*cin + ci (rows padded to 10), byte = (s//2)*32 + o*2 + s%2.
    v4 (hypothesis): s = ky*kw*cin + kx*cin + ci, no row padding, odd tail slot alone."""
    W = _i(L.blobs[0], np.int8); co = L.params[0]; kh, kw = L.params[1], L.params[2]
    per_group = len(W) // (co // 16)
    v5 = blocked(L)
    if not v5:                         # v4: [ky][kx][ci][out] (verified 2026-11-24)
        return W.reshape(kh, kw, cin, co).transpose(3, 0, 1, 2)
    row = 10 if v5 else kw * cin
    nslots = kh * row
    wk = np.zeros((co, kh, kw, cin), np.int64)
    for g in range(co // 16):
        for ky in range(kh):
            for kx in range(kw):
                for ci in range(cin):
                    s = ky * row + kx * cin + ci
                    for o in range(16):
                        b = (s // 2) * 32 + o * 2 + s % 2 if v5 else _pair_byte(s, nslots, o)
                        wk[g * 16 + o, ky, kx, ci] = W[g * per_group + b]
    return wk


def dw_weights(L, c):
    """type 73. v5: per 16 channels 160 bytes; t0 t1 | t2 t3 | t4 t5 | t6 t7 | pad t8.
    v4 (hypothesis): 144 bytes; t0 t1 | t2 t3 | t4 t5 | t6 t7 | t8 alone (16 bytes)"""
    W = _i(L.blobs[0], np.int8)
    wk = np.zeros((c, 9), np.int64)
    if blocked(L):
        pos = [0, 1, 2, 3, 4, 5, 6, 7, 9]
        for g in range(c // 16):
            for t in range(9):
                for o in range(16):
                    wk[g * 16 + o, t] = W[g * 160 + (pos[t] // 2) * 32 + o * 2 + pos[t] % 2]
        return wk
    return W.reshape(9, c).T           # v4: tap-major [ky*3+kx][c] (verified 2026-11-24)


def pw_weights(L, cin):
    """types 53/74: blocks [out group of 16][input pair], each 16 outputs x 2 inputs"""
    W = _i(L.blobs[0], np.int8); co = L.params[0]; ncp = cin // 2
    if not blocked(L):                 # v4 / v5 flag 0: plain [out][in] (verified 2026-11-24)
        return W.reshape(co, cin)
    wk = np.zeros((co, cin), np.int64)
    for og in range(co // 16):
        for cp in range(ncp):
            blk = W[(og * ncp + cp) * 32:(og * ncp + cp + 1) * 32].reshape(16, 2)
            wk[og * 16:og * 16 + 16, cp * 2:cp * 2 + 2] = blk
    return wk


def same_pad(n, k, s):
    """TF 'SAME' padding (before, after) of a length-n axis: output ceil(n / s)"""
    tot = max((-(-n // s) - 1) * s + k - n, 0)
    return tot // 2, tot - tot // 2


def maxpool2(y):
    return np.maximum.reduce([y[i::2, j::2] for i in (0, 1) for j in (0, 1)])


def forward_layer(m, L, ins, bits_of):
    P, t = L.params, L.type
    if t == 16:
        return [ins[0]]
    if t == 46:
        x = ins[0]; H, W_, cin = x.shape
        wk = first_weights(L, cin); sh_, sw_ = P[5], P[6]
        # libjzdl ignores the pad fields P[8:12] and the dilation fields P[3]/P[4] (verified 2026-11-24):
        # always TF "SAME" padding (odd extra at bottom/right), dilate by zero-stuffing the kernel.
        # P[15] = input bits (8 on the image); the padding is input code 0.
        (pt, pb), (pl, pr) = same_pad(H, P[1], sh_), same_pad(W_, P[2], sw_)
        xp = np.pad(x, ((pt, pb), (pl, pr), (0, 0)), constant_values=-(1 << (P[15] - 1)))
        Ho = (xp.shape[0] - P[1]) // sh_ + 1; Wo = (xp.shape[1] - P[2]) // sw_ + 1
        acc = np.zeros((Ho, Wo, P[0]), np.int64)
        for ky in range(P[1]):
            for kx in range(P[2]):
                acc += np.einsum('hwc,oc->hwo', xp[ky:ky + (Ho - 1) * sh_ + 1:sh_, kx:kx + (Wo - 1) * sw_ + 1:sw_], wk[:, ky, kx, :])
        # with weights under 8 bits (P[14]) the sum of products wraps to int16 BEFORE the int32
        # bias is added (verified 2026-11-24, 4-bit); 8-bit weights keep int32 (verified 2026-11-25)
        if P[14] < 8:
            acc = ((acc + 32768) & 0xffff) - 32768
        return [requant(acc, L, 31, P[16])]
    if t in (65, 73):
        # 65 = 8-bit depthwise (Magik 1.0 kit): bits at 15/16, int32 bias/mult/shift, Q31,
        # no int16 wrap (verified 2026-11-25)
        x = ins[0]; c = x.shape[2]; bi, bo = (P[15], P[16]) if t == 65 else (P[14], P[15])
        wk = dw_weights(L, c)
        sh_, sw_ = P[5], P[6]; pt, pb, pl, pr = P[8:12]
        xp = np.pad(x, ((pt, pb), (pl, pr), (0, 0)), constant_values=-(1 << (bi - 1)))
        Ho = (xp.shape[0] - 3) // sh_ + 1; Wo = (xp.shape[1] - 3) // sw_ + 1
        acc = np.zeros((Ho, Wo, c), np.int64)
        for k in range(9):
            ky, kx = divmod(k, 3)
            acc += xp[ky:ky + (Ho - 1) * sh_ + 1:sh_, kx:kx + (Wo - 1) * sw_ + 1:sw_] * wk[:, k]
        return [requant(acc, L, 31 if t == 65 else 15, bo)]
    if t in (52, 53, 74):
        # 52 = 8-bit 1x1 (Magik 1.0 kit): int32 bias/mult/shift, Q31 (verified 2026-11-25)
        x = ins[0]; y = requant(np.einsum('hwc,oc->hwo', x, pw_weights(L, x.shape[2])), L, 31 if t == 52 else 15, P[15])
        return [maxpool2(y) if t == 74 else y]
    if t in (50, 51):
        x = ins[0]; co = P[0]
        wk = _i(L.blobs[0], np.int8).reshape(co, x.shape[2])
        bias = _i(L.blobs[1], np.int32); sc = np.frombuffer(L.blobs[2], np.float32)
        acc = np.einsum('hwc,oc->hwo', x, wk)
        return [((acc + bias).astype(np.float32) * sc).astype(np.float32)]
    if t == 33:
        return [ins[0], ins[0]]
    if t == 75:
        return [maxpool2(ins[0])]
    if t == 69:
        x = ins[0]
        if P[4] == 0:                 # verified: zero insertion (P[1] = input bits)
            z = np.full((x.shape[0] * P[2], x.shape[1] * P[3], x.shape[2]), -(1 << (bits_of[L.bottoms[0]] - 1)), np.int64)
            z[::P[2], ::P[3]] = x
            return [z]
        # P[4] = 1, verified (yolov5, Magik 1.0 kit at 4/5/8 bits): nearest neighbour
        return [np.repeat(np.repeat(x, P[2], axis=0), P[3], axis=1)]
    if t == 78:
        # verified 2026-11-24 (yolov5, bits 5+5 -> 5): mean of the unsigned values,
        # rounded half to even; P[0] = bits_out
        bo = P[0]
        ua = ins[0] + (1 << (bits_of[L.bottoms[0]] - 1)); ub = ins[1] + (1 << (bits_of[L.bottoms[1]] - 1))
        s2 = ua + ub; k = s2 >> 1
        u = k + (s2 & 1) * (k & 1)
        return [np.clip(u, 0, (1 << bo) - 1) - (1 << (bo - 1))]
    if t == 71:
        return [np.concatenate(ins, axis=2)]
    if t == 76:
        parts, trailer = [], []
        for x in ins:
            parts.append(x.reshape(-1)); trailer += [x.shape[0], x.shape[1], x.shape[2]]
        return [np.concatenate(parts + [np.array(trailer + [len(ins)], np.float32)]).astype(np.float32)]
    raise NotImplementedError('layer type %d' % t)


def forward(m, rgb, want=None):
    """rgb: HxWx3 uint8. Returns dict blob -> array (int64 activations / float32 heads)."""
    global LAST_BLOBS
    VERSION[0] = m.version
    blobs = {m.input_blob: rgb.astype(np.int64) - 128}
    LAST_BLOBS = blobs
    bits = {m.input_blob: 8}
    for L in m.layers:
        if L.type == 16:
            continue
        outs = forward_layer(m, L, [blobs[b] for b in L.bottoms], bits)
        for b, o in zip(L.tops, outs):
            blobs[b] = o
            if L.type in (46, 65):
                bits[b] = L.params[16]
            elif L.type in (52, 53, 73, 74):
                bits[b] = L.params[15]
            elif L.type == 78:
                bits[b] = L.params[0]
            elif L.bottoms:
                bits[b] = bits[L.bottoms[0]]
    return blobs


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(2)
    m = jzdl.Model.parse(open(sys.argv[1], 'rb').read())
    rgb = np.fromfile(sys.argv[2], np.uint8).reshape(m.in_h, m.in_w, m.in_c)
    out = forward(m, rgb)[m.output_blob]
    o = out if out.ndim == 3 else out.reshape(1, 1, -1)
    with open(sys.argv[3], 'wb') as f:
        f.write(struct.pack('<3i', o.shape[1], o.shape[0], o.shape[2]))
        f.write(o.astype(np.float32).tobytes())
    print('output blob %d shape %s -> %s' % (m.output_blob, o.shape, sys.argv[3]))
