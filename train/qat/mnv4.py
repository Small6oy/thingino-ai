"""MobileNetV4-conv-small-0.5 backbone + YOLOX-Nano neck and heads, as a T31 JZDL graph.

  arch 'mnv4n', init 'mnv4' (backbone: timm's Apache-2.0 ImageNet weights; neck + heads: Megvii's
  Apache-2.0 YOLOX-Nano COCO weights; three 1x1 adapters start random) or 'scratch'.

What changes from MobileNetV4 (timm mobilenetv4_conv_small_050):
  - 5x5 depthwise kernels -> their centre 3x3 (the T31 has 3x3 depthwise only).
  - Layers without activation (UIR dw_start and pw_proj, and so the residual stream) store their
    value + a per-channel offset (T31 activations are unsigned), measured on calibration images
    with the T31's padding; see mapweights.py. Offsets would add up along the residual stream (to
    hundreds: its 5-bit codes all offset, and the T31's requant limit broken downstream), so each
    residual add is followed by a re-centring layer: a 3x3 depthwise conv with only its centre tap
    (a per-channel scale + bias, ~0.2M MACs a block) that resets the stream to its own offset and
    gain 1. Clipping the stream at 0 instead loses the pretraining (feature cosine 0.07).
  - Stride-2 layers pad bottom/right only (TF 'SAME'), as on the T31.
  - Dense 3x3 convs (stem, blocks 0.0 and 1.0) are JZDL type 46: the T31's slow path (~0.6 GMAC/s),
    so this backbone suits the T41 better than the T31 (see macs()).
Features: x2 = blocks.1.1 (stride 8, 32 ch), x1 = end of stage 2 (stride 16, 48 ch),
x0 = blocks.4.0 (stride 32, 480 ch); adapters to YOLOX's 64 / 128 / 256 channels.
Check: python3 qat/mnv4.py MNV4.safetensors YOLOX.pth (float graph == MobileNetV4 with the
changes above, at the adapters' inputs).
"""
import json, os, struct, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from graphgen import Graph  # noqa: E402
import yoloxn  # noqa: E402

M = 'mnv4.'
# stage 2 and 3 UIR blocks of conv_small_050: (block, cin, cout, dw_start k, exp ch, dw_mid k, stride)
UIR = [('2.0', 32, 48, 5, 96, 5, 2), ('2.1', 48, 48, 0, 96, 3, 1), ('2.2', 48, 48, 0, 96, 3, 1),
       ('2.3', 48, 48, 0, 96, 3, 1), ('2.4', 48, 48, 0, 96, 3, 1), ('2.5', 48, 48, 3, 192, 0, 1),
       ('3.0', 48, 64, 3, 288, 3, 2), ('3.1', 64, 64, 5, 256, 5, 1), ('3.2', 64, 64, 0, 256, 5, 1),
       ('3.3', 64, 64, 0, 192, 5, 1), ('3.4', 64, 64, 0, 256, 3, 1), ('3.5', 64, 64, 0, 256, 3, 1)]
STAGE2_END = '2.5'
BN_EPS = 1e-5


def read_safetensors(path):
    """{name: float32 tensor} without the safetensors package (F32 / F16 / BF16 only)."""
    b = open(path, 'rb').read()
    n = struct.unpack('<Q', b[:8])[0]
    hdr = json.loads(b[8:8 + n]); hdr.pop('__metadata__', None)
    out = {}
    for k, v in hdr.items():
        s, e = v['data_offsets']
        raw = b[8 + n + s:8 + n + e]
        if v['dtype'] == 'F32':
            a = np.frombuffer(raw, '<f4')
        elif v['dtype'] == 'F16':
            a = np.frombuffer(raw, '<f2').astype(np.float32)
        elif v['dtype'] == 'BF16':
            a = (np.frombuffer(raw, '<u2').astype(np.uint32) << 16).view(np.float32)
        elif v['dtype'] == 'I64':
            continue                                   # num_batches_tracked
        else:
            raise ValueError('%s: dtype %s' % (k, v['dtype']))
        out[k] = torch.from_numpy(a.reshape(v['shape']).copy())
    return out


def load_state(mnv4_path, yolox_path=None):
    """timm MobileNetV4 keys -> 'mnv4.<layer>.conv.weight' / '.bn.*', plus YOLOX's keys as they are."""
    sd = {}
    for k, v in read_safetensors(mnv4_path).items():
        k = k.replace('conv_stem.', 'stem.conv.').replace('bn1.', 'bn.')
        if k.startswith('bn.'):
            k = 'stem.' + k
        sd[M + k] = v.float()
    if yolox_path:
        sd.update(yoloxn.load_state(yolox_path))
    return sd


def weights_paths():
    root = os.environ.get('THINGINO_PRETRAINED') or os.path.join(HERE, '..', 'cache', 'pretrained')
    p = os.path.join(root, 'mnv4_small_050', 'model.safetensors')
    if not os.path.exists(p):
        raise SystemExit('MobileNetV4 weights not found: %s (python3 data/fetch_pretrained.py %s --only mnv4_small_050,yolox_nano)' % (p, root))
    return p, yoloxn.weights_path()


def backbone(g):
    y = g.conv('image', 'm.stem', 32, 3, 2, src=M + 'stem')
    y = g.conv(y, 'm.b0.0', 16, 3, 2, src=M + 'blocks.0.0')
    y = g.pw(y, 'm.b0.1', 16, src=M + 'blocks.0.1')
    y = g.conv(y, 'm.b1.0', 48, 3, 2, src=M + 'blocks.1.0')
    x2 = y = g.pw(y, 'm.b1.1', 32, src=M + 'blocks.1.1')
    x1 = None
    for blk, cin, cout, ks, exp, km, s in UIR:
        p, name = M + 'blocks.' + blk, 'm.b' + blk
        t = y
        if ks:
            t = g.dw(t, name + '.ds', s if not km else 1, src=p + '.dw_start', linear=True)
        t = g.pw(t, name + '.pe', exp, src=p + '.pw_exp')
        if km:
            t = g.dw(t, name + '.dm', s, src=p + '.dw_mid')
        skip = cin == cout and s == 1
        t = g.pw(t, name + '.pp', cout, src=p + '.pw_proj', linear=True, match=y if skip else None)
        if skip:                     # re-centre the stream: its own offset, gain 1 (see module doc)
            y = g.dw(g.add(y, t, name), name + '.rc', src=None, linear=True, identity=True)
        else:
            y = t
        if blk == STAGE2_END:
            x1 = y
    x0 = g.pw(y, 'm.b4.0', 480, src=M + 'blocks.4.0')
    a2 = g.pw(x2, 'ad2', 64)                           # new layers: random start
    a1 = g.pw(x1, 'ad1', 128)
    a0 = g.pw(x0, 'ad0', 256)
    return a2, a1, a0


def build(num_classes, in_h, in_w, sd=None, bits=5):
    from graphnet import GraphNet
    g = Graph(bits)
    heads = yoloxn.neck_head(g, sd, *backbone(g), num_classes)
    m, _, _ = g.finish(heads, in_h, in_w)
    net = GraphNet(m, num_classes, in_h, in_w, signed46=True)   # type 46 in mid-graph: its real accumulator rule
    net.src_model, net.graph_nodes = m, g.nodes
    return net


def first_conv(W):
    """timm input (x / 255 - 0.5) / 0.5 -> our x / 255: W' = 2W, b = -sum W."""
    return 2.0 * W, -W.sum((1, 2, 3))


@torch.no_grad()
def measure_offsets(sd, images, margin=1.5):
    """{linear blob: offset}, measured on images (N x 3 x H x W, 0..255) with the T31's padding."""
    offsets = {}
    mnv4_forward(sd, images.float() / 255.0, offsets=offsets, margin=margin)
    return offsets


@torch.no_grad()
def from_mnv4(mnv4_path, yolox_path, classes, in_h, in_w, seed=0, bits=5, sd=None, k_sigma=4.0, calib=None):
    """calib: images (N x 3 x H x W, 0..255) to measure the linear layers' offsets on (else the
    BatchNorm guess, which clips: use calib)."""
    from mapweights import map_weights
    sd = sd if sd is not None else load_state(mnv4_path, yolox_path)
    offsets = measure_offsets(sd, calib) if calib is not None else None
    torch.manual_seed(seed)
    net = build(len(classes), in_h, in_w, sd, bits)
    gen = torch.Generator().manual_seed(seed)
    map_weights(net, sd, first_fn=first_conv, k_sigma=k_sigma, bn_eps=BN_EPS, offsets=offsets,
                head_fn=lambda c, ex: yoloxn._init_head(c, sd, ex['level'], classes, gen))
    _set_eps(net, yoloxn.BN_EPS)                       # YOLOX's neck / heads keep YOLOX's BatchNorm eps
    return net


def _set_eps(net, eps_yolox):
    """BatchNorm eps per source: timm 1e-5, YOLOX 1e-3 (new layers: timm's)."""
    convs = [i for i, s in enumerate(net.spec) if s[0] in (46, 73, 53, 51)]
    for (t, ins, out, P, src, ex), i in zip([n for n in net.graph_nodes if n[0] in (46, 73, 53, 51)], convs):
        c = net.convs[str(i)]
        if c.bn is not None and src is not None and not src.startswith(M):
            c.bn.eps = eps_yolox


@torch.no_grad()
def scratch_float(classes, in_h, in_w, seed=0, bits=5):
    torch.manual_seed(seed)
    net = build(len(classes), in_h, in_w, None, bits)
    per = 5 + len(classes)
    for c in net.convs.values():
        c.quant_w = False
        if c.act is not None:
            c.act.enabled = False
            c.bn, c.bn_folded = nn.BatchNorm2d(c.cout, eps=BN_EPS), False
        else:
            c.bias.view(3, per)[:, 4:] = -4.6
    return net


# ------------------------------------------------------------------ checks
def mnv4_forward(sd, x, go=None, stem=None, offsets=None, margin=1.5):
    """MobileNetV4 with the T31 changes (centre-3x3 depthwise, SAME stride 2) in plain PyTorch,
    x = RGB / 255; returns true-valued (x2, x1, x0). The T31 pads with stored 0, i.e. true -offset,
    so a dw_start's input is padded in the offset domain: offsets from go (graph blob -> (gain,
    offset)), or, with offsets={} to fill, measured on x layer by layer in graph order (each linear
    layer: margin * -(its per-channel minimum), a residual: the sum of its inputs')."""
    stem = stem if stem is not None else first_conv(sd[M + 'stem.conv.weight'])

    def bn(y, p, act):
        y = F.batch_norm(y, sd[p + '.bn.running_mean'], sd[p + '.bn.running_var'], sd[p + '.bn.weight'],
                         sd[p + '.bn.bias'], False, 0.0, BN_EPS)
        return F.relu(y) if act else y

    def conv(y, p, s=1, groups=1, act=True, off=None, W=None, b=None):
        W = sd[p + '.conv.weight'] if W is None else W
        if W.shape[-1] > 3:
            a = (W.shape[-1] - 3) // 2
            W = W[:, :, a:a + 3, a:a + 3]
        k = W.shape[-1]
        if k > 1:
            pads = (0, 1, 0, 1) if s == 2 else (k // 2,) * 4
            if off is None:
                y = F.pad(y, pads)
            else:
                o = off.view(1, -1, 1, 1).to(y.dtype)
                y = F.pad(y + o, pads) - o
        return bn(F.conv2d(y, W, b, s, 0, 1, groups), p, act)

    def off_of(name, y):
        if go is not None:
            return go[name][1]
        o = (-y.amin((0, 2, 3))).clamp_min(0) * margin
        offsets[name] = o
        return o

    y = conv(x, M + 'stem', 2, W=stem[0], b=stem[1])
    y = conv(conv(y, M + 'blocks.0.0', 2), M + 'blocks.0.1')
    x2 = y = conv(conv(y, M + 'blocks.1.0', 2), M + 'blocks.1.1')
    oy = torch.zeros(y.shape[1])
    for blk, cin, cout, ks, exp, km, s in UIR:
        p, name = M + 'blocks.' + blk, 'm.b' + blk
        t = y
        if ks:
            t = conv(t, p + '.dw_start', s if not km else 1, cin, act=False, off=oy)
            off_of(name + '.ds', t)
        t = conv(t, p + '.pw_exp')
        if km:
            t = conv(t, p + '.dw_mid', s, exp)
        t = conv(t, p + '.pw_proj', act=False)
        ot = off_of(name + '.pp', t)
        skip = cin == cout and s == 1
        if skip:
            y = y + t
            oy = off_of(name + '.rc', y)
        else:
            y, oy = t, ot
        if blk == STAGE2_END:
            x1 = y
    x0 = conv(y, M + 'blocks.4.0')
    return x2, x1, x0


@torch.no_grad()
def check(mnv4_path, yolox_path, in_h=320, in_w=416):
    """Float graph == MobileNetV4 (T31 changes) at the adapters' inputs (offsets measured on the test images: no clipping)."""
    sd = load_state(mnv4_path, yolox_path)
    torch.manual_seed(0)
    img = torch.rand(2, 3, in_h, in_w) * 255
    net = from_mnv4(mnv4_path, yolox_path, ['person', 'pet', 'package', 'vehicle'], in_h, in_w, sd=sd, calib=img,
                    ).eval()
    stem = net.convs[str([i for i, s in enumerate(net.spec) if s[0] == 46][0])]
    ref = mnv4_forward(sd, img / 255.0, net.blob_go, (stem.weight, stem.bias))
    names = [n for n in net.graph_nodes if n[0] in (46, 73, 53, 51)]
    idx = [i for i, s in enumerate(net.spec) if s[0] in (46, 73, 53, 51)]
    ad = {n[2]: (str(i), n[1][0]) for n, i in zip(names, idx) if n[2] in ('ad2', 'ad1', 'ad0')}
    got = {}
    hooks = []
    for k, (i, src_blob) in ad.items():
        def hk(m, inp, o, k=k, b=src_blob):
            g, off = net.blob_go[b]
            got[k] = inp[0] * g.view(1, -1, 1, 1).float() - off.view(1, -1, 1, 1).float()
        hooks.append(net.convs[i].register_forward_hook(hk))
    net(img)
    for h in hooks:
        h.remove()
    worst = 0.0
    for k, r in zip(('ad2', 'ad1', 'ad0'), ref):
        e = ((got[k] - r).abs().max() / r.abs().max()).item()
        worst = max(worst, e)
        print('%s input %s: max rel error %.2e' % (k, tuple(r.shape), e))
    return worst


if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) < 2:
        raise SystemExit('usage: mnv4.py MNV4.safetensors YOLOX_NANO.pth')
    err = check(a[0], a[1])
    print('mapping %s (worst %.2e)' % ('OK' if err < 1e-4 else 'WRONG', err))
    net = from_mnv4(a[0], a[1], ['person', 'pet', 'package', 'vehicle'], 320, 416)
    m = yoloxn.macs(net)
    print('MACs per frame %s, total %.1fM, %d layers' % ({k: '%.1fM' % (v / 1e6) for k, v in m.items()},
                                                        sum(m.values()) / 1e6, len(net.spec)))
