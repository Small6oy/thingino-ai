"""TinyDet as a JZDL graph: an anchor-free single-scale detector made only of full 3x3 convs.

TinyDet (opensensor/thingino-accel, training/tinydet.py): 320x192 input, stride 16, one 20x12 grid,
conv+BN+ReLU only:

    3->16 s2, 16->16, 16->32 s2, 32->32, 32->64 s2, 64->64 dil 2, 64->64 s2, 64->64 dil 4,
    head 64->64, 64->32, 1x1 32 -> 5 + classes  (obj, x, y, w, h, classes; anchor-free)

Box decode (their runtime, src/mars/mars_detect.c): every output through a sigmoid;
cx = (gx + sx) * stride, cy = (gy + sy) * stride, w = sw * input width, h = sh * input height,
score = obj * class.

On the T31 every 3x3 conv is a JZDL type 46 (a full conv that libjzdl runs anywhere in the graph,
P[15] = input bits; int16 wrapping sum of w * signed codes, int32 bias after: 2^(b-1) * sum|w| <= 32767
per output, so a 64-channel 3x3 needs narrow activations, e.g. 4-5 bits), the dilated ones as zero-stuffed 5x5 / 9x9 kernels (libjzdl
has no dilation); the 1x1 head is type 51 (float out). See docs/jzdl-format.md. libjzdl pads type 46
TF 'SAME' (stride 2 on even sizes: nothing top/left, one row/column bottom/right), so TinyDet's
stride-2 convs (trained with padding 1 on both sides) see their input one pixel shifted until fine-tuned.

Their weights come as a .mars file (BN folded into the conv bias, weights OHWI); TinyDet's own
repository has no licence file, so models made from those weights stay private.
"""
import os, struct, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..', 'tools', 'jzdl'))
import jzdl  # noqa: E402

DT = 20
IN_H, IN_W, STRIDE = 192, 320, 16
# (cin, cout, stride, dilation) of the ten 3x3 convs
SPEC = [(3, 16, 2, 1), (16, 16, 1, 1), (16, 32, 2, 1), (32, 32, 1, 1), (32, 64, 2, 1), (64, 64, 1, 2),
        (64, 64, 2, 1), (64, 64, 1, 4), (64, 64, 1, 1), (64, 32, 1, 1)]
SRC_CLASSES = ['person', 'vehicle', 'cat', 'dog']           # the .mars checkpoint's head order
# our class -> the source classes whose head rows start it (averaged); None = new class
FROM_SRC = {'person': ['person'], 'pet': ['cat', 'dog'], 'vehicle': ['vehicle']}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # TinyDet's input normalisation (RGB)


def skeleton(num_classes, in_h=IN_H, in_w=IN_W, bits=5, head_bits=5, dilation=True):
    """JZDL v4 model of TinyDet with empty weight blobs. bits: activation bits of every conv
    output; head_bits: of the last conv (the head's input). dilation=False: plain 3x3 instead of the
    two dilated convs (zero-stuffed 5x5 / 9x9 on the T31: about 180M of TinyDet's ~310M MACs)."""
    m = jzdl.Model()
    m.version, m.in_h, m.in_w, m.in_c = 4, in_h, in_w, 3
    layers = [jzdl.Layer(16, [], [0], [3, 67, 67])]
    b_in = 8                                                   # the image
    for k, (cin, cout, s, d) in enumerate(SPEC):
        d = d if dilation else 1
        ke = 2 * d + 1                                         # 3x3 dilated d -> (3-1)*d+1
        bo = head_bits if k == len(SPEC) - 1 else bits
        pt, pb = (0, 1) if s == 2 else (d, d)                  # what libjzdl does anyway ('SAME', even sizes)
        P = [cout, ke, ke, d, d, s, s, -233, pt, pb, pt, pb, 0, 0, 4, b_in, bo, DT]      # v4: 18 params
        layers.append(jzdl.Layer(46, [k], [k + 1], P))
        b_in = bo
    n = len(SPEC)
    layers.append(jzdl.Layer(51, [n], [n + 1], [5 + num_classes, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, b_in, 32, 21]))
    layers.append(jzdl.Layer(76, [n + 1], [n + 2], []))
    m.layers = layers
    m.input_blob, m.output_blob, m.blob_count = 0, n + 2, n + 3
    return m


def build(num_classes, in_h=IN_H, in_w=IN_W, bits=5, head_bits=5, dilation=True):
    """(GraphNet, its JZDL skeleton), type 46 bounded by its own accumulator rule."""
    from graphnet import GraphNet
    sk = skeleton(num_classes, in_h, in_w, bits, head_bits, dilation)
    net = GraphNet(sk, num_classes, in_h, in_w, anchors_per_head=1, signed46=True)
    net.src_model = sk
    return net, sk


def read_mars(path):
    """{tensor name: float32 array} of a .mars file (thingino-accel include/mars.h, MIT):
    64+12-byte header, 124-byte tensor records, 112-byte layer records, then the weights."""
    b = open(path, 'rb').read()
    magic, _, _, _, nl, nt, ni, no, woff, wsz = struct.unpack_from('<IHHIIIIIQQ', b, 0)
    if magic != 0x5352414D:
        raise ValueError('%s: not a .mars file' % path)
    out, p = {}, 76
    for _ in range(nt):
        name = b[p + 4:p + 64].split(b'\0')[0].decode()
        dt, fmt, nd = struct.unpack_from('<III', b, p + 64)
        shp = struct.unpack_from('<6i', b, p + 76)[:nd]
        off, sz = struct.unpack_from('<QQ', b, p + 100)
        if sz:
            if dt != 0:
                raise ValueError('%s: tensor %s is not float32' % (path, name))
            a = np.frombuffer(b, '<f4', sz // 4, woff + off)
            if fmt == 6 and len(shp) == 4:                     # OHWI, shape listed as O, I, H, W
                o, i, kh, kw = shp
                a = a.reshape(o, kh, kw, i).transpose(0, 3, 1, 2)
            out[name] = np.ascontiguousarray(a.reshape(shp))
        p += 124
    return out


@torch.no_grad()
def from_mars(path, classes, in_h=IN_H, in_w=IN_W, seed=0, **kw):
    """GraphNet (float: activations off) holding TinyDet's .mars weights. The ImageNet input
    normalisation is folded into the first conv (the graph's input is plain 0..255 RGB).
    The head rows are mapped to our classes (FROM_SRC); a class TinyDet lacks starts small."""
    t = read_mars(path)
    ws = [k for k in t if k.endswith('weight')]
    if len(ws) != len(SPEC) + 1:
        raise ValueError('%s: %d weight tensors, TinyDet has %d' % (path, len(ws), len(SPEC) + 1))
    net, sk = build(len(classes), in_h, in_w, **kw)
    convs = [net.convs[str(i)] for i, L in enumerate(sk.layers) if L.type in (46, 51)]
    g = torch.Generator().manual_seed(seed)
    for c, wn in zip(convs, ws):
        W = torch.from_numpy(t[wn]).double(); B = torch.from_numpy(t[wn[:-len('weight')] + 'bias']).double()
        if c is convs[0]:
            # their input (x/255 - mean) / std; ours x/255 (GraphNet in_scale): W' = W/std, B' = B - sum W mean/std
            sd = torch.tensor(STD, dtype=torch.float64).view(1, 3, 1, 1); mu = torch.tensor(MEAN, dtype=torch.float64).view(1, 3, 1, 1)
            B = B - (W * mu / sd).sum((1, 2, 3)); W = W / sd
        if c is convs[-1]:
            Wn = torch.zeros(5 + len(classes), *W.shape[1:], dtype=torch.float64); Bn = torch.zeros(5 + len(classes), dtype=torch.float64)
            Wn[:5], Bn[:5] = W[:5], B[:5]
            std = W[5:].std()
            for k, name in enumerate(classes):
                src = [5 + SRC_CLASSES.index(s) for s in FROM_SRC.get(name, [])]
                if src:
                    Wn[5 + k], Bn[5 + k] = W[src].mean(0), B[src].mean(0)
                else:
                    Wn[5 + k] = torch.randn(W.shape[1:], generator=g, dtype=torch.float64) * std * 0.1; Bn[5 + k] = -4.6
            W, B = Wn, Bn
        if c.weight.shape != W.shape:
            raise ValueError('weight shape %s != %s' % (tuple(W.shape), tuple(c.weight.shape)))
        c.weight.copy_(W.float()); c.bias.copy_(B.float())
        c.quant_w = False
        if c.act is not None:
            c.act.enabled = False
    return net


def decode(out, in_h, in_w, conf=0.05, nms_iou=0.45, max_det=100):
    """out: N x (5 + C) x gh x gw logits -> per image (k, 6) tensors x0 y0 x1 y1 score cls (pixels)."""
    from torchvision.ops import batched_nms
    n, _, gh, gw = out.shape
    p = out.float().sigmoid()
    gy, gx = torch.meshgrid(torch.arange(gh, device=out.device), torch.arange(gw, device=out.device), indexing='ij')
    cx = (gx + p[:, 1]) * in_w / gw; cy = (gy + p[:, 2]) * in_h / gh
    bw = p[:, 3] * in_w; bh = p[:, 4] * in_h
    sc, c = p[:, 5:].max(1)
    score = p[:, 0] * sc
    d = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, score, c.float()], -1).view(n, -1, 6)
    res = []
    for e in d:
        e = e[e[:, 4] > conf]
        k = batched_nms(e[:, :4], e[:, 4], e[:, 5], nms_iou)
        res.append(e[k[:max_det]].cpu())
    return res
