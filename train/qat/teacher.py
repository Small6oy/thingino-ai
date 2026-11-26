"""A bigger float teacher for distillation: YOLOX (Nano .. L) as Megvii released it, with our heads.

  arch 'yoloxn' student, --distill yolox_s (or yolox_tiny, yolox_m, yolox_l, yolox_nano).

The teacher keeps everything the T31 student cannot have (SiLU, the SPP pools, the Focus stem,
full width and depth), so it is the accurate model the student learns from. Only the heads are
ours: YOLOv5-style anchors on strides 8/16/32 fed by both YOLOX branches (cls + reg features
concatenated), started from YOLOX's own obj / cls predictions exactly as yoloxn.from_yolox does.
Same heads, same anchors, same grid as the student, so distillation compares the two cell by cell
(train.kd_loss).

The teacher is never exported or run on a camera; it lives only during training. It is trained in
float on our classes first (build_model.py: the teacher stage), then frozen.

Weights: data/fetch_pretrained.py yolox_<size> (Apache-2.0). Checks: python3 qat/teacher.py --check
"""
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import yoloxn  # noqa: E402  (load_state, csp_depth, _init_head, BN_EPS, prefixes)

SIZES = ('yolox_nano', 'yolox_tiny', 'yolox_s', 'yolox_m', 'yolox_l')
B, P = yoloxn.B, yoloxn.P


def weights_path(size):
    root = os.environ.get('THINGINO_PRETRAINED') or os.path.join(HERE, '..', 'cache', 'pretrained')
    p = os.path.join(root, size, size + '.pth')
    if not os.path.exists(p):
        raise SystemExit('%s weights not found: %s (python3 data/fetch_pretrained.py %s --only %s)' % (size, p, root, size))
    return p


class Conv(nn.Module):
    """YOLOX BaseConv: conv (no bias) + BatchNorm + SiLU, from the state dict under prefix p."""

    def __init__(self, sd, p, s=1):
        super().__init__()
        w = sd[p + '.conv.weight']
        o, i, k, _ = w.shape
        g = o if i == 1 else 1                             # depthwise (DWConv's dconv): in == out channels
        self.conv = nn.Conv2d(i * g, o, k, s, (k - 1) // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(o, eps=yoloxn.BN_EPS, momentum=0.03)
        with torch.no_grad():
            self.conv.weight.copy_(w)
            for n in ('weight', 'bias', 'running_mean', 'running_var'):
                getattr(self.bn, n).copy_(sd[p + '.bn.' + n])

    def forward(self, x):
        return F.silu(self.bn(self.conv(x)))


class DW(nn.Module):
    """YOLOX DWConv: 3x3 depthwise + 1x1."""

    def __init__(self, sd, p, s=1):
        super().__init__()
        self.dconv, self.pconv = Conv(sd, p + '.dconv', s), Conv(sd, p + '.pconv')

    def forward(self, x):
        return self.pconv(self.dconv(x))


def unit(sd, p, s=1):
    """A 3x3 YOLOX conv: DWConv in the depthwise models (Nano), BaseConv in the others."""
    return DW(sd, p, s) if p + '.dconv.conv.weight' in sd else Conv(sd, p, s)


class CSP(nn.Module):
    def __init__(self, sd, p, shortcut):
        super().__init__()
        self.conv1, self.conv2, self.conv3 = Conv(sd, p + '.conv1'), Conv(sd, p + '.conv2'), Conv(sd, p + '.conv3')
        n = yoloxn.csp_depth(sd, p)
        self.m1 = nn.ModuleList(Conv(sd, '%s.m.%d.conv1' % (p, i)) for i in range(n))
        self.m2 = nn.ModuleList(unit(sd, '%s.m.%d.conv2' % (p, i)) for i in range(n))
        self.shortcut = shortcut

    def forward(self, x):
        a = self.conv1(x)
        for c1, c2 in zip(self.m1, self.m2):
            b = c2(c1(a))
            a = a + b if self.shortcut else b
        return self.conv3(torch.cat([a, self.conv2(x)], 1))


class Teacher(nn.Module):
    """YOLOX as released (float, SiLU, SPP, Focus) + our anchor heads. Input: RGB 0..255, as the student (GraphNet)."""

    def __init__(self, sd, classes, anchors, seed=0):
        super().__init__()
        self.stem = Conv(sd, B + 'stem.conv')
        self.down = nn.ModuleList(unit(sd, B + 'dark%d.0' % i, 2) for i in (2, 3, 4, 5))
        self.dark = nn.ModuleList(CSP(sd, B + 'dark%d.1' % i, True) for i in (2, 3, 4))
        self.spp1, self.spp2 = Conv(sd, B + 'dark5.1.conv1'), Conv(sd, B + 'dark5.1.conv2')
        self.dark5 = CSP(sd, B + 'dark5.2', False)
        self.lat0, self.c3_p4 = Conv(sd, P + 'lateral_conv0'), CSP(sd, P + 'C3_p4', False)
        self.red1, self.c3_p3 = Conv(sd, P + 'reduce_conv1'), CSP(sd, P + 'C3_p3', False)
        self.bu2, self.c3_n3 = unit(sd, P + 'bu_conv2', 2), CSP(sd, P + 'C3_n3', False)
        self.bu1, self.c3_n4 = unit(sd, P + 'bu_conv1', 2), CSP(sd, P + 'C3_n4', False)
        h = 'head.'
        self.hstem = nn.ModuleList(Conv(sd, h + 'stems.%d' % i) for i in range(3))
        self.cls = nn.ModuleList(nn.Sequential(unit(sd, h + 'cls_convs.%d.0' % i), unit(sd, h + 'cls_convs.%d.1' % i)) for i in range(3))
        self.reg = nn.ModuleList(nn.Sequential(unit(sd, h + 'reg_convs.%d.0' % i), unit(sd, h + 'reg_convs.%d.1' % i)) for i in range(3))
        half = sd[h + 'cls_preds.0.weight'].shape[1]
        per = 5 + len(classes)
        self.heads = nn.ModuleList(nn.Conv2d(2 * half, 3 * per, 1) for _ in range(3))
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for i, c in enumerate(self.heads):
                yoloxn._init_head(c, sd, i, list(classes), gen)
        # YOLOX's own prediction convs, kept only for check() (not trained, not used in forward)
        self._released = {(n, i): (sd[h + '%s.%d.weight' % (n, i)], sd[h + '%s.%d.bias' % (n, i)])
                          for n in ('cls_preds', 'reg_preds', 'obj_preds') for i in range(3)}
        self.strides = [8, 16, 32]
        self.anchors = {s: anchors[s] for s in self.strides}
        self.num_classes = len(classes)

    def features(self, x):
        """-> per level (cls feature, reg feature), strides 8, 16, 32."""
        x = x.flip(1)                                      # our RGB 0..255 -> Megvii's BGR 0..255
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        y = self.stem(x)
        outs = []
        for i in range(3):
            y = self.dark[i](self.down[i](y))
            outs.append(y)
        x2, x1 = outs[1], outs[2]
        y = self.spp1(self.down[3](y))
        y = self.spp2(torch.cat([y] + [F.max_pool2d(y, k, 1, k // 2) for k in (5, 9, 13)], 1))
        x0 = self.dark5(y)
        up = lambda t: F.interpolate(t, scale_factor=2, mode='nearest')
        fpn0 = self.lat0(x0)
        f0 = self.c3_p4(torch.cat([up(fpn0), x1], 1))
        fpn1 = self.red1(f0)
        pan2 = self.c3_p3(torch.cat([up(fpn1), x2], 1))
        pan1 = self.c3_n3(torch.cat([self.bu2(pan2), fpn1], 1))
        pan0 = self.c3_n4(torch.cat([self.bu1(pan1), fpn0], 1))
        feats = []
        for i, f in enumerate((pan2, pan1, pan0)):
            s = self.hstem[i](f)
            feats.append((self.cls[i](s), self.reg[i](s)))
        return feats

    def forward(self, x):
        return [h(torch.cat([c, r], 1)) for h, (c, r) in zip(self.heads, self.features(x))]

    @torch.no_grad()
    def released(self, x):
        """YOLOX's own outputs, in yoloxn.yolox_forward's format (for checks only)."""
        out = []
        for i, (c, r) in enumerate(self.features(x)):
            pr = lambda t, n: F.conv2d(t, *(p.to(t.device, t.dtype) for p in self._released[(n, i)]))
            out.append((c, r, pr(c, 'cls_preds'), pr(r, 'reg_preds'), pr(r, 'obj_preds')))
        return out


def build(size_or_path, classes, anchors, seed=0):
    """Teacher from a size name (SIZES) or a Megvii .pth path."""
    path = weights_path(size_or_path) if size_or_path in SIZES else size_or_path
    t = Teacher(yoloxn.load_state(path), classes, anchors, seed)
    t.source = os.path.basename(path)
    return t


def macs(net, in_h=320, in_w=416):
    """Multiply-accumulates of one forward pass (convs only)."""
    tot = [0]
    was_training = net.training
    dev = next(net.parameters()).device

    def hook(m, i, o):
        tot[0] += o.numel() // o.shape[0] * m.weight[0].numel()
    hs = [m.register_forward_hook(hook) for m in net.modules() if isinstance(m, nn.Conv2d)]
    try:
        with torch.no_grad():
            net.eval()(torch.zeros(1, 3, in_h, in_w, device=dev))
    finally:
        for h in hs:
            h.remove()
        net.train(was_training)
    return tot[0]


def check(root=None):
    """Nano: our module == yoloxn.yolox_forward (the mapping the student uses) on random input.
    Every size: parameter count, MACs at 416x320, head shapes."""
    torch.manual_seed(0)
    anchors = {8: [(10, 13), (16, 30), (33, 23)], 16: [(30, 61), (62, 45), (59, 119)], 32: [(116, 90), (156, 198), (373, 326)]}
    classes = ['person', 'pet', 'vehicle']
    x = torch.rand(2, 3, 320, 416) * 255.0
    for size in SIZES:
        try:
            p = weights_path(size)
        except SystemExit as e:
            print('%-11s skipped: %s' % (size, e)); continue
        sd = yoloxn.load_state(p)
        t = Teacher(sd, classes, anchors).eval()
        with torch.no_grad():
            outs = t(x)
            if size == 'yolox_nano':
                ref = yoloxn.yolox_forward(sd, x.flip(1))
                d = max(float((m - r).abs().max()) for (c, rg), lv in zip(t.features(x), ref) for m, r in ((c, lv[0]), (rg, lv[1])))
                print('yolox_nano  features vs yoloxn.yolox_forward: max diff %.2e' % d)
        n = sum(p_.numel() for p_ in t.parameters())
        print('%-11s %5.2fM params, %6.2f GMACs at 416x320, heads %s' % (size, n / 1e6, macs(t) / 1e9, [tuple(o.shape) for o in outs]))


if __name__ == '__main__':
    if '--check' in sys.argv:
        check()
