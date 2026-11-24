#!/usr/bin/env python3
"""Make a probe model that exposes an intermediate int8 blob.

  probe.py MODEL.bin BLOB OUT.bin [--bits N] [--scale S]

net->run(Mat<float>&, blob) only returns float blobs (the type-51 heads). This
keeps the layers needed to produce BLOB, then attaches a type-51 head on BLOB
with identity weights (C x C), bias 0 and scale S (default 1.0), and makes that
head the model output. Running the probe returns BLOB's values as floats.
"""
import struct, sys
import jzdl


def blob_channels(m, blob):
    """Channel count of a blob, walking producers."""
    for L in m.layers:
        if blob in L.tops:
            if L.type in (46, 51, 53, 73, 74):
                return L.params[0]
            if L.type == 71:
                return sum(blob_channels(m, b) for b in L.bottoms)
            if L.type in (33, 69, 75, 78):
                return blob_channels(m, L.bottoms[0])
            if L.type == 16:
                return m.in_c
    raise ValueError('no producer for blob %d' % blob)


def make_probe(m, blob, bits=None, scale=1.0, weight=1):
    need, keep = {blob}, []
    for L in reversed(m.layers):            # keep only the producers BLOB depends on
        if need & set(L.tops):
            keep.append(L)
            need |= set(L.bottoms)
    keep.reverse()
    c = blob_channels(m, blob)
    if bits is None:                        # head's bits_in: copy the producer's output bits
        prod = keep[-1]
        bits = prod.params[15] if prod.type in (53, 73, 74) else 4
    out = m.blob_count
    w = bytes((weight & 0xff) if i == j else 0 for i in range(c) for j in range(c))
    head = jzdl.Layer(51, [blob], [out],
                      [c, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, c * c, bits, 32, 21] + ([0] if m.version == 5 else []),
                      [w, struct.pack('<%di' % c, *([0] * c)), struct.pack('<%df' % c, *([scale] * c))])
    p = jzdl.Model()
    p.__dict__.update({k: v for k, v in m.__dict__.items() if k != 'layers'})
    p.layers = keep + [head]
    p.blob_count = m.blob_count + 1
    p.output_blob = out
    return p


if __name__ == '__main__':
    a = sys.argv
    if len(a) < 4:
        print(__doc__); sys.exit(2)
    m = jzdl.Model.parse(open(a[1], 'rb').read())
    bits = int(a[a.index('--bits') + 1]) if '--bits' in a else None
    scale = float(a[a.index('--scale') + 1]) if '--scale' in a else 1.0
    p = make_probe(m, int(a[2]), bits, scale)
    open(a[3], 'wb').write(p.to_bytes())
    print('probe for blob %d: %d layers, %d channels -> %s' % (int(a[2]), len(p.layers), p.layers[-1].params[0], a[3]))
