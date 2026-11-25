"""Build JZDL v4 graphs for our own architectures (no template file).

A Graph collects layers by name; finish() returns a jzdl.Model that GraphNet trains and
export_graph() fills. It inserts the split layers (type 33, 1 -> 2) libjzdl needs wherever one blob
feeds more than one layer, and keeps every weight layer's source key (net.src_keys[i]) so a
loader can map pretrained tensors onto it.

Layer parameter layouts (JZDL v4, as the yolov5 sample uses them; docs/jzdl-format.md):
  46 full k x k conv  [cout, k, k, d, d, s, s, -233, pt, pb, pl, pr, wbytes, 0, 4, bits_in, bits_out, 20]
  73 depthwise 3x3    [c, 3, 3, 1, 1, s, s, 0, pt, pb, pl, pr, wbytes, 0, bits_in, bits_out, 20]
  53 1x1              [cout, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, wbytes, 0, bits_in, bits_out, 20]
  51 head 1x1 (float) [cout, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, wbytes, bits_in, 32, 21]
  78 residual mean    [bits_out, 20]      71 concat [20]      33 split [20]
  69 upsample x2      [20, 4, 2, 2, 1]    (f4 = 1: nearest neighbour)      76 output []
Pads: libjzdl uses TF 'SAME' for 46 whatever the fields say; 73 stride 2 pads bottom/right only.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'tools', 'jzdl'))
import jzdl  # noqa: E402

DT = 20


class Graph:
    def __init__(self, bits=5):
        self.bits = bits
        self.nodes = []              # (type, [input names], output name, params, src key, extra)
        self.width = {'image': 3}
        self.bits_of = {'image': 8}
        self.defined = {'image'}

    def _add(self, t, ins, out, P, src=None, **extra):
        if out in self.defined:
            raise ValueError('blob %s defined twice' % out)
        self.defined.add(out)
        self.nodes.append((t, list(ins), out, P, src, extra))
        return out

    def conv(self, x, out, cout, k, s=1, src=None, **extra):
        """full k x k conv (type 46), on the image or anywhere (slow on the T31: ~0.6 GMAC/s)"""
        bi, bo = self.bits_of[x], self.bits
        pt = (k - s) // 2 if k > s else 0
        P = [cout, k, k, 1, 1, s, s, -233, pt, k - s - pt, pt, k - s - pt, 0, 0, 4, bi, bo, DT]
        self.width[out], self.bits_of[out] = cout, bo
        return self._add(46, [x], out, P, src, **extra)

    def dw(self, x, out, s=1, src=None, **extra):
        c, bi, bo = self.width[x], self.bits_of[x], self.bits
        pads = [0, 1, 0, 1] if s == 2 else [1, 1, 1, 1]
        P = [c, 3, 3, 1, 1, s, s, 0] + pads + [0, 0, bi, bo, DT]
        self.width[out], self.bits_of[out] = c, bo
        return self._add(73, [x], out, P, src, **extra)

    def pw(self, x, out, cout, src=None, **extra):
        bi, bo = self.bits_of[x], self.bits
        P = [cout, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, bi, bo, DT]
        self.width[out], self.bits_of[out] = cout, bo
        return self._add(53, [x], out, P, src, **extra)

    def head(self, x, out, cout, src=None, **extra):
        P = [cout, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, self.bits_of[x], 32, 21]
        self.width[out], self.bits_of[out] = cout, None
        return self._add(51, [x], out, P, src, **extra)

    def add(self, a, b, out):
        """residual: libjzdl stores the MEAN of the two inputs (a = the skip path)"""
        if self.width[a] != self.width[b]:
            raise ValueError('add %s: widths %d != %d' % (out, self.width[a], self.width[b]))
        self.width[out], self.bits_of[out] = self.width[a], self.bits
        return self._add(78, [a, b], out, [self.bits, DT])

    def concat(self, xs, out):
        self.width[out], self.bits_of[out] = sum(self.width[x] for x in xs), self.bits_of[xs[0]]
        return self._add(71, xs, out, [DT])

    def up(self, x, out):
        self.width[out], self.bits_of[out] = self.width[x], self.bits_of[x]
        return self._add(69, [x], out, [DT, 4, 2, 2, 1])

    def finish(self, outputs, in_h, in_w):
        """jzdl.Model (version 4) + {layer index: source key} + {layer index: extra}."""
        uses = {}
        for t, ins, out, P, src, ex in self.nodes:
            for x in ins:
                uses[x] = uses.get(x, 0) + 1
        for x in outputs:
            uses[x] = uses.get(x, 0) + 1
        ids = {}
        nxt = [0]

        def new():
            nxt[0] += 1
            return nxt[0] - 1
        layers = [jzdl.Layer(16, [], [new()], [3, 67, 67])]
        ids['image'] = [layers[0].tops[0]]
        src_keys, extras = {}, {}

        def define(name, blob):
            """blob = the producer's output; split it into uses[name] copies."""
            n = uses.get(name, 0)
            copies = [blob]
            while len(copies) < n:
                b = copies.pop()
                t1, t2 = new(), new()
                layers.append(jzdl.Layer(33, [b], [t1, t2], [DT]))
                copies += [t1, t2]
            ids[name] = copies

        def take(name):
            if not ids.get(name):
                raise ValueError('blob %s used more often than counted' % name)
            return ids[name].pop(0)
        define('image', ids.pop('image')[0])
        for t, ins, out, P, src, ex in self.nodes:
            bots = [take(x) for x in ins]
            top = new()
            layers.append(jzdl.Layer(t, bots, [top], list(P)))
            if src is not None:
                src_keys[len(layers) - 1] = src
            if ex:
                extras[len(layers) - 1] = ex
            define(out, top)
        o = new()
        layers.append(jzdl.Layer(76, [take(x) for x in outputs], [o], []))
        m = jzdl.Model()
        m.version, m.in_h, m.in_w, m.in_c = 4, in_h, in_w, 3
        m.layers, m.input_blob, m.output_blob, m.blob_count = layers, 0, o, o + 1
        return m, src_keys, extras
