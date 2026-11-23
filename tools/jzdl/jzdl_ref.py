#!/usr/bin/env python3
"""Bit-exact NumPy reference of libjzdl 5.4.0 (T31) inference for version-5 models.

  jzdl_ref.py MODEL.bin INPUT.rgb OUT.f32     # same input/output format as jzdlrun

Verified 2026-11-23 against libjzdl running under QEMU (fork, -cpu XBurstR2):
magik_model_persondet.bin, every blob and the packed output, 100 % equal on two
frames. See ../../jzdl-format.md "Arithmetic".

Activations are int8 holding u - 2^(bits-1), u = unsigned bits-wide ReLU output.
Requant (per output channel):  u = clamp(rsr(((acc + bias) * mult) >> Q, shift), 0, 2^bits - 1)
  Q = 31 for the first conv (int32 bias/mult/shift), 15 for the others (int16),
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
    return np.clip(rsr(((acc + bias) * mult) >> q, sh), 0, (1 << bits) - 1) - (1 << (bits - 1))


def first_weights(L, cin):
    """type 46: tap s = ky*10 + kx*cin + ci (each kernel row padded to 10), byte = (s//2)*32 + o*2 + s%2 per 16 outputs"""
    W = _i(L.blobs[0], np.int8); co = L.params[0]; kh, kw = L.params[1], L.params[2]
    per_group = len(W) // (co // 16)
    wk = np.zeros((co, kh, kw, cin), np.int64)
    for g in range(co // 16):
        for ky in range(kh):
            for kx in range(kw):
                for ci in range(cin):
                    s = ky * 10 + kx * cin + ci
                    for o in range(16):
                        wk[g * 16 + o, ky, kx, ci] = W[g * per_group + (s // 2) * 32 + o * 2 + s % 2]
    return wk


def dw_weights(L, c):
    """type 73: per 16 channels 160 bytes; tap order t0 t1 | t2 t3 | t4 t5 | t6 t7 | pad t8"""
    W = _i(L.blobs[0], np.int8); pos = [0, 1, 2, 3, 4, 5, 6, 7, 9]
    wk = np.zeros((c, 9), np.int64)
    for g in range(c // 16):
        for t in range(9):
            for o in range(16):
                wk[g * 16 + o, t] = W[g * 160 + (pos[t] // 2) * 32 + o * 2 + pos[t] % 2]
    return wk


def pw_weights(L, cin):
    """types 53/74: blocks [out group of 16][input pair], each 16 outputs x 2 inputs"""
    W = _i(L.blobs[0], np.int8); co = L.params[0]; ncp = cin // 2
    wk = np.zeros((co, cin), np.int64)
    for og in range(co // 16):
        for cp in range(ncp):
            blk = W[(og * ncp + cp) * 32:(og * ncp + cp + 1) * 32].reshape(16, 2)
            wk[og * 16:og * 16 + 16, cp * 2:cp * 2 + 2] = blk
    return wk


def maxpool2(y):
    return np.maximum.reduce([y[i::2, j::2] for i in (0, 1) for j in (0, 1)])


def forward_layer(m, L, ins, bits_of):
    P, t = L.params, L.type
    if t == 16:
        return [ins[0]]
    if t == 46:
        x = ins[0]; H, W_, cin = x.shape
        wk = first_weights(L, cin); sh_, sw_ = P[5], P[6]; pt, pb, pl, pr = P[8:12]
        xp = np.pad(x, ((pt, pb), (pl, pr), (0, 0)), constant_values=-128)
        Ho = (xp.shape[0] - P[1]) // sh_ + 1; Wo = (xp.shape[1] - P[2]) // sw_ + 1
        acc = np.zeros((Ho, Wo, P[0]), np.int64)
        for ky in range(P[1]):
            for kx in range(P[2]):
                acc += np.einsum('hwc,oc->hwo', xp[ky:ky + (Ho - 1) * sh_ + 1:sh_, kx:kx + (Wo - 1) * sw_ + 1:sw_], wk[:, ky, kx, :])
        return [requant(acc, L, 31, P[16])]
    if t == 73:
        x = ins[0]; c = x.shape[2]; bi, bo = P[14], P[15]; wk = dw_weights(L, c)
        sh_, sw_ = P[5], P[6]; pt, pb, pl, pr = P[8:12]
        xp = np.pad(x, ((pt, pb), (pl, pr), (0, 0)), constant_values=-(1 << (bi - 1)))
        Ho = (xp.shape[0] - 3) // sh_ + 1; Wo = (xp.shape[1] - 3) // sw_ + 1
        acc = np.zeros((Ho, Wo, c), np.int64)
        for k in range(9):
            ky, kx = divmod(k, 3)
            acc += xp[ky:ky + (Ho - 1) * sh_ + 1:sh_, kx:kx + (Wo - 1) * sw_ + 1:sw_] * wk[:, k]
        return [requant(acc, L, 15, bo)]
    if t in (53, 74):
        x = ins[0]; y = requant(np.einsum('hwc,oc->hwo', x, pw_weights(L, x.shape[2])), L, 15, P[15])
        return [maxpool2(y) if t == 74 else y]
    if t == 51:
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
        if P[1] != 4:
            raise NotImplementedError('upsample mode %d (only 4 = zero insertion verified)' % P[1])
        x = ins[0]; z = np.full((x.shape[0] * P[2], x.shape[1] * P[3], x.shape[2]), x.min() if False else -(1 << (bits_of[L.bottoms[0]] - 1)), np.int64)
        z[::P[2], ::P[3]] = x
        return [z]
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
    blobs = {m.input_blob: rgb.astype(np.int64) - 128}
    bits = {m.input_blob: 8}
    for L in m.layers:
        if L.type == 16:
            continue
        outs = forward_layer(m, L, [blobs[b] for b in L.bottoms], bits)
        for b, o in zip(L.tops, outs):
            blobs[b] = o
            if L.type in (46,):
                bits[b] = L.params[16]
            elif L.type in (53, 73, 74):
                bits[b] = L.params[15]
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
