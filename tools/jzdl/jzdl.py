#!/usr/bin/env python3
"""JZDL model (.bin) parser / writer for Ingenic libjzdl 5.4.0 (T31, MXU1).

Layout confirmed by running libjzdl.m.so's own loader under emulation
(re/emu.py) and tracing every fread; see ../../jzdl-format.md.

File = header + param section + model section, all little-endian:

  int32 param_bytes        # bytes from offset 24 to the end of the param section
                           # (the in-memory loader parses from buf+24 and checks this)
  int32 input_blob         # Net+0x38, get_model_input_index()
  int32 output_blob        # Net+0x3c, get_model_output_index()
  int32 in_h, in_w, in_c   # input shape (persondet 320,416,3)
  int32 layers | version<<24
  int32 blob_count
  per layer: int32 type, n_bottom, n_top, bottom[n_bottom], top[n_top], params[...]
  then, per layer in order, the layer's model blobs (weights etc.)

The number of param words per layer depends only on (type, version); every
branch in the load_param readers tests the version and nothing else.
"""
import struct, sys

# --- per-type layout --------------------------------------------------------
# param word counts, keyed by type -> {version: count}
PARAM_COUNT = {
    16: {4: 3, 5: 3},            # Input
    33: {4: 1, 5: 1},            # Split
    46: {4: 18, 5: 19},          # Conv, uint8 image input (first layer)
    50: {5: 17},                 # Conv 1x1 head, 8-bit (Magik 1.0 kit Txx YOLOv5, 8-bit)
    52: {5: 18},                 # Conv 1x1, 8-bit, int32 requant (Magik 1.0 kit, 8-bit)
    65: {5: 19},                 # Conv 3x3 depthwise, 8-bit, int32 requant (Magik 1.0 kit, 8-bit)
    51: {4: 16, 5: 17},          # Conv 1x1, int32/float output (detection head)
    53: {4: 17, 5: 18},          # Conv 1x1 (pointwise)
    69: {4: 5, 5: 5},            # Upsample
    71: {4: 1, 5: 1},            # Concat
    73: {4: 17, 5: 18},          # Conv 3x3 depthwise
    74: {5: 18},                 # Conv 1x1 (pointwise, small-channel variant)
    75: {5: 12},                 # Pooling (max 2x2/2)
    76: {4: 0, 5: 0},            # Output pack (heads -> one Mat)
    78: {4: 2, 5: 2},            # Eltwise add (yolov5 residual)
}

# param field names (index -> name). Unnamed fields keep their index.
_CONV_DW = ['num_output', 'kernel_h', 'kernel_w', 'dilation_h', 'dilation_w',
            'stride_h', 'stride_w', 'pad_mode', 'pad_top', 'pad_bottom', 'pad_left', 'pad_right',
            'weight_bytes', 'f13', 'bits_a', 'bits_b', 'dtype', 'v5_flag']
_CONV_PW = ['num_output', 'f1', 'kernel_h', 'kernel_w', 'f4', 'f5',
            'f6', 'f7', 'f8', 'f9', 'f10', 'f11',
            'weight_bytes', 'f13', 'bits_a', 'bits_b', 'dtype', 'v5_flag']
FIELD_NAMES = {
    16: ['f0', 'f1', 'f2'],
    33: ['dtype'], 71: ['dtype'],
    46: ['num_output', 'kernel_h', 'kernel_w', 'dilation_h', 'dilation_w',
         'stride_h', 'stride_w', 'pad_mode', 'pad_top', 'pad_bottom', 'pad_left', 'pad_right',
         'weight_bytes', 'f13', 'bits_w', 'bits_in', 'bits_out', 'dtype', 'v5_flag'],
    73: _CONV_DW,
    53: _CONV_PW, 74: _CONV_PW, 52: _CONV_PW,
    65: _CONV_DW[:14] + ['bits_w', 'bits_in', 'bits_out', 'dtype', 'v5_flag'],
    50: ['num_output', 'f1', 'kernel_h', 'kernel_w', 'f4', 'f5',
         'f6', 'f7', 'f8', 'f9', 'f10', 'f11',
         'weight_bytes', 'bits_in', 'bits_out', 'dtype', 'v5_flag'],
    51: ['num_output', 'f1', 'kernel_h', 'kernel_w', 'f4', 'f5',
         'f6', 'f7', 'f8', 'f9', 'f10', 'f11',
         'weight_bytes', 'bits_in', 'bits_out', 'dtype', 'v5_flag'],
    69: ['dtype', 'bits', 'scale_h', 'scale_w', 'mode'],
    75: ['f0', 'kernel_h', 'kernel_w', 'stride_h', 'stride_w',
         'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'dtype'],
    76: [],
    78: ['f0', 'dtype'],
}


def blob_sizes(ltype, params):
    """Byte sizes of the model blobs a layer reads in load_model, in order."""
    if ltype in (46, 50, 51, 52, 53, 65, 73, 74):
        c = params[0]
        wb = params[12]
        if ltype in (46, 52, 65):
            return [wb, 4 * c, 4 * c, 4 * c]          # weights + 3 x int32[c]
        if ltype in (50, 51):
            return [wb, 4 * c, 4 * c]                 # weights + int32 bias[c] + float32 scale[c]
        return [wb, 2 * c, 2 * c, 2 * c, 2 * c]       # weights + 4 x int16[c]
    return []


class Layer:
    def __init__(self, ltype, bottoms, tops, params, blobs=None):
        self.type, self.bottoms, self.tops, self.params = ltype, list(bottoms), list(tops), list(params)
        self.blobs = blobs if blobs is not None else []

    def field(self, name):
        return self.params[FIELD_NAMES[self.type].index(name)]

    def named(self):
        names = FIELD_NAMES.get(self.type, [])
        return {(names[i] if i < len(names) else 'f%d' % i): v for i, v in enumerate(self.params)}

    def __repr__(self):
        return 'Layer(%d, %s -> %s, %s, blobs=%s)' % (self.type, self.bottoms, self.tops, self.params,
                                                       [len(b) for b in self.blobs])


class Model:
    def __init__(self):
        self.input_blob = 0
        self.output_blob = 0
        self.in_h = self.in_w = self.in_c = 0
        self.version = 5
        self.blob_count = 0
        self.layers = []

    # ---------------------------------------------------------------- parse
    @classmethod
    def parse(cls, data):
        m = cls()
        pos = [0]

        def i32():
            v = struct.unpack_from('<i', data, pos[0])[0]
            pos[0] += 4
            return v

        param_bytes = i32()
        m.input_blob, m.output_blob = i32(), i32()
        m.in_h, m.in_w, m.in_c = i32(), i32(), i32()
        w0 = i32() & 0xffffffff
        nlayers, m.version = w0 & 0xffffff, w0 >> 24
        m.blob_count = i32() & 0xffffff
        for _ in range(nlayers):
            t, nb, nt = i32(), i32(), i32()
            b = [i32() for _ in range(nb)]
            tp = [i32() for _ in range(nt)]
            try:
                n = PARAM_COUNT[t][m.version]
            except KeyError:
                raise ValueError('layer type %d not supported for version %d (at byte %d)'
                                 % (t, m.version, pos[0]))
            m.layers.append(Layer(t, b, tp, [i32() for _ in range(n)]))
        if pos[0] - 24 != param_bytes:
            raise ValueError('param section ends at %d, header says %d+24' % (pos[0], param_bytes))
        for L in m.layers:
            for sz in blob_sizes(L.type, L.params):
                L.blobs.append(bytes(data[pos[0]:pos[0] + sz]))
                if len(L.blobs[-1]) != sz:
                    raise ValueError('truncated model data')
                pos[0] += sz
        if pos[0] != len(data):
            raise ValueError('%d trailing bytes after model data' % (len(data) - pos[0]))
        return m

    # ---------------------------------------------------------------- write
    def param_words(self):
        w = [0, self.input_blob, self.output_blob, self.in_h, self.in_w, self.in_c,
             (len(self.layers) & 0xffffff) | (self.version << 24), self.blob_count]
        for L in self.layers:
            if len(L.params) != PARAM_COUNT[L.type][self.version]:
                raise ValueError('%r: expected %d params' % (L, PARAM_COUNT[L.type][self.version]))
            w += [L.type, len(L.bottoms), len(L.tops)] + L.bottoms + L.tops + L.params
        w[0] = 4 * len(w) - 24
        return w

    def to_bytes(self):
        w = self.param_words()
        out = bytearray(struct.pack('<%dI' % len(w), *[x & 0xffffffff for x in w]))
        for L in self.layers:
            want = blob_sizes(L.type, L.params)
            if [len(b) for b in L.blobs] != want:
                raise ValueError('%r: blob sizes %s, expected %s' % (L, [len(b) for b in L.blobs], want))
            for b in L.blobs:
                out += b
        return bytes(out)

    def summary(self):
        s = ['version %d  input blob %d -> output blob %d  input %dx%dx%d (HxWxC)  %d layers, %d blobs'
             % (self.version, self.input_blob, self.output_blob, self.in_h, self.in_w, self.in_c,
                len(self.layers), self.blob_count)]
        for i, L in enumerate(self.layers):
            s.append('%3d t%-3d %-10s -> %-8s %s  blobs=%s' % (
                i, L.type, ','.join(map(str, L.bottoms)), ','.join(map(str, L.tops)),
                ' '.join(map(str, L.params)), [len(b) for b in L.blobs]))
        return '\n'.join(s)


def main(argv):
    if len(argv) < 2:
        print('usage: jzdl.py MODEL.bin [--roundtrip OUT.bin]')
        return 2
    data = open(argv[1], 'rb').read()
    m = Model.parse(data)
    print(m.summary())
    out = m.to_bytes()
    print('roundtrip byte-identical:', out == data)
    if '--roundtrip' in argv:
        open(argv[argv.index('--roundtrip') + 1], 'wb').write(out)
    return 0 if out == data else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
