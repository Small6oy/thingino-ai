"""YOLOX-Nano as a T31 JZDL graph, started from Megvii's Apache-2.0 COCO weights.

  arch 'yoloxn', init 'yolox' (weights: data/fetch_pretrained.py yolox_nano) or 'scratch'.

What changes from YOLOX-Nano (Megvii-BaseDetection/YOLOX, depth 0.33, width 0.25, depthwise):
  - activations: SiLU -> ReLU (the T31's unsigned codes); the float stage re-learns around it.
  - stem: Focus + 3x3 is exactly a 6x6 stride-2 conv on the image (stem='focus6'), but on the T31
    that is ~57M MACs on the slow type 46 path and its int16 accumulator bound leaves ~3-bit
    weights. Default stem='conv3': a 3x3 stride-2 conv (as persondet / YOLO-N), started from the
    6x6's centre taps.
  - SPP (5/9/13 max pools, stride 1) has no T31 layer: the pools become identity, i.e. conv2's four
    input groups are summed into one 1x1.
  - heads: ours, YOLOv5-style anchors on strides 8/16/32 (raptor-hal's decode), fed by both YOLOX
    branches (cls + reg features concatenated). Objectness and class rows start from YOLOX's
    obj / cls predictions (vehicle = mean of car, motorcycle, bus, truck; pet = cat, dog); box rows
    start at zero (= the anchor at the cell centre); a class YOLOX lacks (package) starts small.
  - residuals: libjzdl's type 78 stores the MEAN of its inputs. The loader tracks a per-channel gain
    (true value / stored value) through the graph and folds it into the next layer's weights, so
    the float graph equals the ReLU version of YOLOX exactly (check: python3 qat/yoloxn.py --check).

The float reference yolox_forward() is only used to check this mapping.
"""
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from graphgen import Graph  # noqa: E402

# Input of Megvii's released checkpoints (checked on our val: 0.59 vs 0.01 for normalised RGB): BGR pixels 0..255,
# no normalisation. Our graph's input is RGB / 255, so the first conv takes W' = 255 * W[:, ::-1].
BN_EPS = 1e-3
COCO = {'person': [0], 'pet': [15, 16], 'vehicle': [2, 3, 5, 7]}   # our class -> COCO class ids (YOLOX head rows)
B = 'backbone.backbone.'
P = 'backbone.'


# ------------------------------------------------------------------ weights
def load_state(path):
    """YOLOX checkpoint -> {name: tensor} (weights_only: the .pth is a pickle)."""
    ck = torch.load(path, map_location='cpu', weights_only=True)
    sd = ck.get('model', ck)
    return {k: v.float() for k, v in sd.items() if hasattr(v, 'shape')}


DEPTH = {B + 'dark2.1': 1, B + 'dark3.1': 3, B + 'dark4.1': 3}      # YOLOX-Nano's CSP depths (others: 1)


def csp_depth(sd, prefix):
    if sd is None:
        return DEPTH.get(prefix, 1)
    return len({k.split('.m.')[1].split('.')[0] for k in sd if k.startswith(prefix + '.m.')})


def weights_path():
    """data/fetch_pretrained.py yolox_nano puts it here (or set THINGINO_PRETRAINED to OUT)."""
    root = os.environ.get('THINGINO_PRETRAINED') or os.path.join(HERE, '..', 'cache', 'pretrained')
    p = os.path.join(root, 'yolox_nano', 'yolox_nano.pth')
    if not os.path.exists(p):
        raise SystemExit('YOLOX-Nano weights not found: %s (python3 data/fetch_pretrained.py %s --only yolox_nano)' % (p, root))
    return p


def focus_to_6x6(W):
    """Focus (space-to-depth 2x2, channel groups TL, BL, TR, BR) + k x k conv == 2k x 2k stride-2 conv."""
    o, c4, k, _ = W.shape
    c = c4 // 4
    W6 = W.new_zeros(o, c, 2 * k, 2 * k)
    for g, (dy, dx) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        W6[:, :, dy::2, dx::2] = W[:, g * c:(g + 1) * c]
    return W6


# ------------------------------------------------------------------ float reference (checks only)
def yolox_forward(sd, x, act='silu', spp=True, stem=None, s2pad='torch'):
    """YOLOX-Nano in plain PyTorch from its state dict. x: N x 3 x H x W, already normalised (or,
    with stem=(W, b), raw RGB/255 fed to that stem conv with TF SAME padding).
    s2pad='same': stride-2 depthwise padded bottom/right only, as the T31 (type 73 pads 0 1 0 1);
    'torch': YOLOX's own padding 1 on every side.
    Returns per level (cls_feat, reg_feat, cls_logits, reg, obj_logits), strides 8, 16, 32."""
    A = F.silu if act == 'silu' else F.relu

    def bn(y, p):
        return F.batch_norm(y, sd[p + '.bn.running_mean'], sd[p + '.bn.running_var'], sd[p + '.bn.weight'],
                            sd[p + '.bn.bias'], False, 0.0, BN_EPS)

    def base(y, p, k=1, s=1, groups=1):
        return A(bn(F.conv2d(y, sd[p + '.conv.weight'], None, s, (k - 1) // 2, 1, groups), p))

    def dwconv(y, p, s=1):
        if s == 2 and s2pad == 'same':
            q = p + '.dconv'
            d = A(bn(F.conv2d(F.pad(y, (0, 1, 0, 1)), sd[q + '.conv.weight'], None, 2, 0, 1, y.shape[1]), q))
        else:
            d = base(y, p + '.dconv', 3, s, y.shape[1])
        return base(d, p + '.pconv')

    def csp(y, p, shortcut):
        a = base(y, p + '.conv1')
        for i in range(csp_depth(sd, p)):
            q = '%s.m.%d' % (p, i)
            b = dwconv(base(a, q + '.conv1'), q + '.conv2')
            a = a + b if shortcut else b
        return base(torch.cat([a, base(y, p + '.conv2')], 1), p + '.conv3')

    if stem is None:
        f = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        y = base(f, B + 'stem.conv', 3)
    else:
        from net import same_pad
        W, b = stem
        (pt, pb), (pl, pr) = (same_pad(n, W.shape[-1], 2) for n in x.shape[2:])
        y = A(bn(F.conv2d(F.pad(x, (pl, pr, pt, pb)), W, b, 2), B + 'stem.conv'))
    y = csp(dwconv(y, B + 'dark2.0', 2), B + 'dark2.1', True)
    x2 = y = csp(dwconv(y, B + 'dark3.0', 2), B + 'dark3.1', True)
    x1 = y = csp(dwconv(y, B + 'dark4.0', 2), B + 'dark4.1', True)
    y = dwconv(y, B + 'dark5.0', 2)
    y = base(y, B + 'dark5.1.conv1')
    pools = [F.max_pool2d(y, k, 1, k // 2) for k in (5, 9, 13)] if spp else [y, y, y]
    y = base(torch.cat([y] + pools, 1), B + 'dark5.1.conv2')
    x0 = csp(y, B + 'dark5.2', False)
    up = lambda t: F.interpolate(t, scale_factor=2, mode='nearest')
    fpn0 = base(x0, P + 'lateral_conv0')
    f0 = csp(torch.cat([up(fpn0), x1], 1), P + 'C3_p4', False)
    fpn1 = base(f0, P + 'reduce_conv1')
    pan2 = csp(torch.cat([up(fpn1), x2], 1), P + 'C3_p3', False)
    pan1 = csp(torch.cat([dwconv(pan2, P + 'bu_conv2', 2), fpn1], 1), P + 'C3_n3', False)
    pan0 = csp(torch.cat([dwconv(pan1, P + 'bu_conv1', 2), fpn0], 1), P + 'C3_n4', False)
    out = []
    for i, f in enumerate((pan2, pan1, pan0)):
        h = 'head.'
        s = base(f, h + 'stems.%d' % i)
        c = dwconv(dwconv(s, h + 'cls_convs.%d.0' % i), h + 'cls_convs.%d.1' % i)
        r = dwconv(dwconv(s, h + 'reg_convs.%d.0' % i), h + 'reg_convs.%d.1' % i)
        pred = lambda t, n: F.conv2d(t, sd[h + '%s.%d.weight' % (n, i)], sd[h + '%s.%d.bias' % (n, i)])
        out.append((c, r, pred(c, 'cls_preds'), pred(r, 'reg_preds'), pred(r, 'obj_preds')))
    return out


# ------------------------------------------------------------------ the T31 graph
def _dwconv(g, x, name, p, cout, s=1, **ex):
    """YOLOX DWConv: 3x3 depthwise + 1x1 (src None: new layers)."""
    d = g.dw(x, name + '.d', s, src=p and p + '.dconv')
    return g.pw(d, name, cout, src=p and p + '.pconv', **ex)


def _csp(g, sd, x, name, p, cout, shortcut):
    h = cout // 2
    a = g.pw(x, name + '.c1', h, src=p + '.conv1')
    for i in range(csp_depth(sd, p)):
        q = '%s.m.%d' % (p, i)
        t = g.pw(a, '%s.m%d.c1' % (name, i), h, src=q + '.conv1')
        b = _dwconv(g, t, '%s.m%d.c2' % (name, i), q + '.conv2', h, match=a if shortcut else None)
        a = g.add(a, b, '%s.m%d' % (name, i)) if shortcut else b
    c2 = g.pw(x, name + '.c2', h, src=p + '.conv2')
    return g.pw(g.concat([a, c2], name + '.cat'), name, cout, src=p + '.conv3')


def backbone(g, sd, stem='conv3'):
    """YOLOX-Nano's CSPDarknet: (x2, x1, x0) at strides 8 / 16 / 32, 64 / 128 / 256 channels."""
    y = g.conv('image', 'stem', 16, 6 if stem == 'focus6' else 3, 2, src=B + 'stem.conv')
    y = _csp(g, sd, _dwconv(g, y, 'd2', B + 'dark2.0', 32, 2), 'd2c', B + 'dark2.1', 32, True)
    x2 = _csp(g, sd, _dwconv(g, y, 'd3', B + 'dark3.0', 64, 2), 'd3c', B + 'dark3.1', 64, True)
    x1 = _csp(g, sd, _dwconv(g, x2, 'd4', B + 'dark4.0', 128, 2), 'd4c', B + 'dark4.1', 128, True)
    y = _dwconv(g, x1, 'd5', B + 'dark5.0', 256, 2)
    y = g.pw(y, 'spp.c1', 128, src=B + 'dark5.1.conv1')
    y = g.pw(y, 'spp.c2', 256, src=B + 'dark5.1.conv2', spp=4)
    x0 = _csp(g, sd, y, 'd5c', B + 'dark5.2', 256, False)
    return x2, x1, x0


def neck_head(g, sd, x2, x1, x0, num_classes):
    """YOLOX-Nano's PAFPN (inputs 64 / 128 / 256 channels) + our heads; returns the head blobs."""
    fpn0 = g.pw(x0, 'lat0', 128, src=P + 'lateral_conv0')
    f0 = _csp(g, sd, g.concat([g.up(fpn0, 'lat0.up'), x1], 'p4.in'), 'p4', P + 'C3_p4', 128, False)
    fpn1 = g.pw(f0, 'red1', 64, src=P + 'reduce_conv1')
    pan2 = _csp(g, sd, g.concat([g.up(fpn1, 'red1.up'), x2], 'p3.in'), 'p3', P + 'C3_p3', 64, False)
    pan1 = _csp(g, sd, g.concat([_dwconv(g, pan2, 'bu2', P + 'bu_conv2', 64, 2), fpn1], 'n3.in'), 'n3', P + 'C3_n3', 128, False)
    pan0 = _csp(g, sd, g.concat([_dwconv(g, pan1, 'bu1', P + 'bu_conv1', 128, 2), fpn0], 'n4.in'), 'n4', P + 'C3_n4', 256, False)
    heads = []
    for i, f in enumerate((pan2, pan1, pan0)):
        h = 'head.'
        st = g.pw(f, 'h%d.stem' % i, 64, src=h + 'stems.%d' % i)
        c = _dwconv(g, _dwconv(g, st, 'h%d.c0' % i, h + 'cls_convs.%d.0' % i, 64), 'h%d.c1' % i, h + 'cls_convs.%d.1' % i, 64)
        r = _dwconv(g, _dwconv(g, st, 'h%d.r0' % i, h + 'reg_convs.%d.0' % i, 64), 'h%d.r1' % i, h + 'reg_convs.%d.1' % i, 64)
        heads.append(g.head(g.concat([c, r], 'h%d.cat' % i), 'h%d' % i, 3 * (5 + num_classes), level=i))
    return heads


def build(num_classes, in_h, in_w, sd=None, stem='conv3', bits=5):
    """GraphNet in its QAT form (no BatchNorm), e.g. to load a trained state into."""
    from graphnet import GraphNet
    g = Graph(bits)
    heads = neck_head(g, sd, *backbone(g, sd, stem), num_classes)
    m, _, _ = g.finish(heads, in_h, in_w)
    net = GraphNet(m, num_classes, in_h, in_w)
    net.src_model, net.graph_nodes = m, g.nodes
    return net


def first_conv(W, k):
    """Megvii's Focus stem weights -> our k x k stride-2 image conv on RGB / 255 (+ bias)."""
    W6 = focus_to_6x6(W)
    if k == 3:                                         # centre taps, same total |weight|
        c = W6[:, :, 2:5, 2:5]
        W6 = c * (W6.abs().sum((1, 2, 3), keepdim=True) / c.abs().sum((1, 2, 3), keepdim=True).clamp_min(1e-9))
    return 255.0 * W6.flip(1), torch.zeros(W.shape[0], dtype=W.dtype)    # BGR 0..255 -> RGB / 255


@torch.no_grad()
def from_yolox(path, classes, in_h, in_w, stem='conv3', seed=0, bits=5, sd=None):
    """Float GraphNet (activations off, BatchNorm on, eps as YOLOX) holding YOLOX-Nano's weights.
    Ready for the float stage (train.pretrain_float), then make_qat_init + QAT."""
    from mapweights import map_weights
    sd = sd if sd is not None else load_state(path)
    net = build(len(classes), in_h, in_w, sd, stem, bits)
    gen = torch.Generator().manual_seed(seed)
    k = 6 if stem == 'focus6' else 3
    map_weights(net, sd, first_fn=lambda W: first_conv(W, k),
                head_fn=lambda c, ex: _init_head(c, sd, ex['level'], classes, gen), bn_eps=BN_EPS)
    return net


@torch.no_grad()
def scratch_float(classes, in_h, in_w, stem='conv3', seed=0, bits=5):
    """Random YOLOX-Nano-shaped float GraphNet (BatchNorm on, activations off) for float training."""
    torch.manual_seed(seed)
    net = build(len(classes), in_h, in_w, None, stem, bits)
    per = 5 + len(classes)
    for c in net.convs.values():
        c.quant_w = False
        if c.act is not None:
            c.act.enabled = False
            c.bn, c.bn_folded = nn.BatchNorm2d(c.cout, eps=BN_EPS), False
        else:                                              # head: objectness + class priors ~ 0.01
            c.bias.view(3, per)[:, 4:] = -4.6
    return net


def _init_head(c, sd, i, classes, gen):
    """Our head rows (3 anchors x [x y w h obj cls...]) from YOLOX's obj / cls predictions."""
    h = 'head.'
    Wc, Bc = sd[h + 'cls_preds.%d.weight' % i][:, :, 0, 0], sd[h + 'cls_preds.%d.bias' % i]
    Wo, Bo = sd[h + 'obj_preds.%d.weight' % i][0, :, 0, 0], sd[h + 'obj_preds.%d.bias' % i][0]
    half = Wc.shape[1]                                     # input = [cls feat | reg feat]
    per = 5 + len(classes)
    W = torch.zeros(3 * per, 2 * half); Bv = torch.zeros(3 * per)
    for a in range(3):
        r = a * per
        W[r + 4, half:], Bv[r + 4] = Wo, Bo
        for k, name in enumerate(classes):
            src = COCO.get(name)
            if src:
                W[r + 5 + k, :half], Bv[r + 5 + k] = Wc[src].mean(0), Bc[src].mean()
            else:
                W[r + 5 + k, :half] = torch.randn(half, generator=gen) * Wc.std() * 0.1; Bv[r + 5 + k] = -4.6
    c.weight.copy_(W.view(3 * per, 2 * half, 1, 1)); c.bias.copy_(Bv)


# ------------------------------------------------------------------ checks
@torch.no_grad()
def check(path, in_h=320, in_w=416, classes=('person', 'pet', 'package', 'vehicle')):
    """The float graph (focus6 stem) == YOLOX with ReLU and identity SPP, at the heads' inputs."""
    sd = load_state(path)
    net = from_yolox(path, list(classes), in_h, in_w, stem='focus6', sd=sd).eval()
    torch.manual_seed(0)
    img = torch.rand(2, 3, in_h, in_w) * 255
    c46 = net.convs[str([i for i, s in enumerate(net.spec) if s[0] == 46][0])]
    W, b = c46.folded()
    ref = yolox_forward(sd, img / 255.0, act='relu', spp=False, stem=(c46.weight, c46.bias), s2pad='same')
    feats = {}
    hooks = [net.convs[str(i)].register_forward_hook(lambda m, inp, o, i=i: feats.setdefault(i, inp[0]))
             for i, s in enumerate(net.spec) if s[0] == 51]
    outs = net(img)
    for h in hooks:
        h.remove()
    worst = 0.0
    for lvl, i in enumerate(sorted(feats)):
        cr = torch.cat([ref[lvl][0], ref[lvl][1]], 1)
        e = ((feats[i] - cr).abs().max() / cr.abs().max()).item()
        worst = max(worst, e)
        print('level %d (stride %d): head input %s, max rel error %.2e' % (lvl, in_h // cr.shape[2], tuple(cr.shape), e))
    print('heads:', [tuple(o.shape) for o in outs], 'strides', net.strides)
    return worst


def macs(net):
    """multiply-accumulates per frame of each weight layer kind (T31 cost estimate)."""
    tot = {}
    hooks = []
    for i, c in net.convs.items():
        def h(m, inp, o, kind=c.kind):
            k = m.weight[0].numel()
            tot[kind] = tot.get(kind, 0) + o.shape[1] * o.shape[2] * o.shape[3] * k
        hooks.append(c.register_forward_hook(h))
    with torch.no_grad():
        net.eval()(torch.zeros(1, 3, net.in_h, net.in_w))
    for h in hooks:
        h.remove()
    return tot


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('weights', help='yolox_nano.pth (data/fetch_pretrained.py)')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    err = check(a.weights)
    print('mapping %s (worst %.2e)' % ('OK' if err < 1e-4 else 'WRONG', err))
    for stem in ('focus6', 'conv3'):
        net = from_yolox(a.weights, ['person', 'pet', 'package', 'vehicle'], 320, 416, stem=stem)
        m = macs(net)
        print('stem %s: MACs per frame %s, total %.1fM, %d layers' % (stem, {k: '%.1fM' % (v / 1e6) for k, v in m.items()},
                                                                      sum(m.values()) / 1e6, len(net.spec)))
