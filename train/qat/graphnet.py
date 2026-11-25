"""A trainable, exact twin of any JZDL graph built from the persondet / yolov5 layer set.

GraphNet reads a JZDL model (jzdl.Model) for its topology, mirrors every layer in PyTorch with the
same fake quantisation as net.py (bit-exact with libjzdl), and can start from the model's own
weights. export_graph() writes the trained weights back into the same graph, so the result runs on
the T31 like the original.

Layer types (docs/jzdl-format.md): 16 input, 46 full conv (the first layer, or k x k anywhere), 73 depthwise, 53 1x1, 74 1x1 + maxpool,
51 head (float out), 33 split, 75 maxpool 2x2, 69 upsample (f4 = 0 zero insertion, 1 nearest),
71 concat, 78 eltwise mean (residual), 76 output.

Scales: every activation is u * s with a learned per-tensor s. Type 78 averages the two inputs'
unsigned codes, which is only a real-valued mean if both inputs share one scale, so the inputs and
output of every 78 are tied to one scale (union-find over blobs); split / upsample / maxpool pass
their input's scale through; a concat output carries a per-channel scale vector.

Type 46 anywhere: P[15] = input bits, P[1]/P[2] = kernel, P[3]/P[4] = dilation d (ignored by libjzdl,
which runs the stored zero-stuffed (k-1)*d+1 kernel; GraphNet trains the k x k taps with dilation d).
libjzdl also ignores 46's pad fields: always TF 'SAME' padding (odd extra at the bottom/right).
signed46: bound type 46 by its real accumulator rule (int16 sum of w * signed input codes, int32 bias
added after; net.weight_step) instead of the generic one persondet/yolon were built with (bias inside). anchors_per_head = 1: an anchor-free head (TinyDet), 5 + classes per cell.
"""
import copy, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..', 'tools', 'jzdl'))
import jzdl, jzdl_ref  # noqa: E402
from net import QConv, ActQ, ste_round, zero_insert_up, f64_device, _HAS_TRITON  # noqa: E402

if _HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _t78_fwd_kernel(a_ptr, b_ptr, s_ptr, out_ptr, u_ptr, m_ptr, n_el, max_q, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_el
        s = tl.load(s_ptr)
        ua = tl.extra.cuda.libdevice.rint(tl.load(a_ptr + offs, mask=mask, other=0.0) / s)
        ub = tl.extra.cuda.libdevice.rint(tl.load(b_ptr + offs, mask=mask, other=0.0) / s)
        u_r = tl.extra.cuda.libdevice.rint((ua + ub) * 0.5)
        u_c = tl.minimum(tl.maximum(u_r, 0.0), max_q)
        in_b = (u_r > 0.0) & (u_r < max_q)
        tl.store(out_ptr + offs, u_c * s, mask=mask)
        tl.store(u_ptr + offs, u_c.to(tl.uint8), mask=mask)
        tl.store(m_ptr + offs, in_b.to(tl.uint8), mask=mask)

    @triton.jit
    def _t78_bwd_kernel(go_ptr, a_ptr, b_ptr, u_ptr, m_ptr, s_ptr, ga_ptr, gb_ptr, gs_p, n_el, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_el
        s = tl.load(s_ptr)
        go = tl.load(go_ptr + offs, mask=mask, other=0.0)
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        u = tl.load(u_ptr + offs, mask=mask, other=0).to(tl.float32)
        in_b = tl.load(m_ptr + offs, mask=mask, other=0) != 0
        g_ab = tl.where(in_b, 0.5 * go, 0.0)
        gs_el = tl.where(mask, go * tl.where(in_b, u - (a + b) / (2.0 * s), u), 0.0)
        tl.store(ga_ptr + offs, g_ab, mask=mask)
        tl.store(gb_ptr + offs, g_ab, mask=mask)
        tl.store(gs_p + pid, tl.sum(gs_el, axis=0))

    class _FusedType78(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b, s, bits):
            ac, bc = a.contiguous(), b.contiguous()
            n_el = ac.numel()
            out = torch.empty_like(ac)
            if n_el == 0:
                empty_u8 = torch.empty(0, dtype=torch.uint8, device=ac.device)
                ctx.save_for_backward(ac, bc, empty_u8, empty_u8, s)
                return out
            ut = torch.empty(ac.shape, dtype=torch.uint8, device=ac.device)
            mt = torch.empty(ac.shape, dtype=torch.uint8, device=ac.device)
            _t78_fwd_kernel[((n_el + 1023) // 1024,)](ac, bc, s, out, ut, mt, n_el, float((1 << bits) - 1), BLOCK=1024)
            ctx.save_for_backward(ac, bc, ut, mt, s)
            return out

        @staticmethod
        def backward(ctx, go):
            ac, bc, ut, mt, s = ctx.saved_tensors
            goc = go.contiguous()
            n_el = goc.numel()
            ga, gb = torch.empty_like(goc), torch.empty_like(goc)
            if n_el > 0:
                nb = (n_el + 1023) // 1024
                gsp = torch.empty((nb,), dtype=torch.float32, device=goc.device)
                _t78_bwd_kernel[(nb,)](goc, ac, bc, ut, mt, s, ga, gb, gsp, n_el, BLOCK=1024)
                gs = gsp.sum()
            else:
                gs = torch.zeros_like(s)
            return ga, gb, gs, None

KIND = {46: 'first', 73: 'dw', 53: 'pw', 74: 'pwpool', 51: 'head'}
STOCK_ANCHORS = {8: [(10, 13), (16, 30), (33, 23)], 16: [(30, 61), (62, 45), (59, 119)],
                 32: [(116, 90), (156, 198), (373, 326)]}


def _dil(P):
    """(dilation, k x k taps trained) of a type 46 layer"""
    d = max(1, P[3])
    return d, (P[1] - 1) // d + 1


def _bits(L):
    """(bits_in, bits_out) of a weight layer from its JZDL params."""
    P = L.params
    if L.type == 46:
        return P[15], P[16]
    if L.type == 51:
        return P[13], None
    return P[14], P[15]


class GraphNet(nn.Module):
    def __init__(self, model, num_classes, in_h, in_w, anchors_per_head=3, signed46=False):
        super().__init__()
        self.spec = [(L.type, list(L.bottoms), list(L.tops), list(L.params)) for L in model.layers]
        self.version, self.in_h, self.in_w = model.version, in_h, in_w
        self.num_classes, self.in_scale = num_classes, 1.0 / 255.0
        self.na = anchors_per_head
        self.no = anchors_per_head * (5 + num_classes)
        # union-find: blobs that must share an activation scale
        parent = {}

        def find(b):
            while parent.setdefault(b, b) != b:
                b = parent[b]
            return b

        def union(a, b):
            parent[find(a)] = find(b)
        for t, bot, top, P in self.spec:
            if t in (33, 69, 75):
                for x in top:
                    union(x, bot[0])
            elif t == 78:
                union(bot[0], bot[1]); union(top[0], bot[0])
        self.convs = nn.ModuleDict()
        self.scales = nn.ParameterDict()
        self.group = {}
        widths = {}
        for i, (t, bot, top, P) in enumerate(self.spec):
            if t == 16:
                widths[top[0]] = 3
            elif t in KIND:
                bi, bo = _bits(model.layers[i])
                cin = widths[bot[0]]
                cout = self.no if t == 51 else P[0]
                d, k = _dil(P) if t == 46 else (1, 3)
                c = QConv(KIND[t], cin, cout, bi, bo, k=k)
                c.conv_stride = P[5] if t in (46, 73) else 1
                c.conv_pad = (P[10], P[11], P[8], P[9]) if t in (46, 73) else (0, 0, 0, 0)   # l, r, t, b
                c.conv_dil = d
                c.same_pad = t == 46                             # libjzdl ignores 46's pad fields
                c.acc_signed = t == 46 and signed46
                c.bn, c.bn_folded = None, True
                if c.act is not None:
                    g = 'g%d' % find(top[0])
                    if g not in self.scales:
                        self.scales[g] = nn.Parameter(torch.zeros(()))
                    c.act.log_s = self.scales[g]                 # shared across tied blobs
                self.convs[str(i)] = c
                widths[top[0]] = cout
            elif t == 33:
                for x in top:
                    widths[x] = widths[bot[0]]
            elif t in (69, 75, 78):
                widths[top[0]] = widths[bot[0]]
            elif t == 71:
                widths[top[0]] = sum(widths[b] for b in bot)
        self.out_layer = [i for i, s in enumerate(self.spec) if s[0] == 76][0]
        head_tops = self.spec[self.out_layer][1]
        self.head_layers = [i for t2 in head_tops for i, s in enumerate(self.spec) if s[0] == 51 and s[2][0] == t2]
        self._scale_keys = list(self.scales.keys())
        _sk_idx = {k: idx for idx, k in enumerate(self._scale_keys)}
        self._conv_scale_idx = {}
        for i, (t, bot, top, P) in enumerate(self.spec):
            if t in KIND and self.convs[str(i)].act is not None:
                self._conv_scale_idx[str(i)] = _sk_idx['g%d' % find(top[0])]
        self.register_buffer('_in_scale_t', torch.tensor(self.in_scale, dtype=torch.float32), persistent=False)
        # stride of each head from the graph geometry (in order of the output layer)
        self.strides = self._head_strides()
        self.anchor_free = anchors_per_head == 1
        self.anchors = ({s: STOCK_ANCHORS[s] for s in self.strides}
                        if not self.anchor_free and set(self.strides) <= set(STOCK_ANCHORS) else None)
        self.bits_out78 = {i: P[0] for i, (t, b, tp, P) in enumerate(self.spec) if t == 78}

    def _head_strides(self):
        with torch.no_grad():
            outs = self.forward(torch.zeros(1, 3, self.in_h, self.in_w), _shapes_only=True)
        return [self.in_h // o.shape[2] for o in outs]

    def _scale(self, blob_scale, b):
        return blob_scale[b]

    def forward(self, img, _shapes_only=False):
        """img: N x 3 x H x W, pixel values 0..255. Returns the head logits (NCHW) in output order."""
        x0 = img * self.in_scale
        act_on = (not _shapes_only) and any(c.act is not None and c.act.enabled for c in self.convs.values())
        scales_vec = None
        if act_on and self._scale_keys:
            ls = torch.stack([self.scales[k] for k in self._scale_keys])
            fd = f64_device(ls.device)
            scales_vec = ls.exp() if fd is None else ls.to(fd).double().exp().to(ls.dtype).to(ls.device)
        val, sc = {}, {}
        for i, (t, bot, top, P) in enumerate(self.spec):
            if t == 16:
                val[top[0]] = x0
                if act_on:
                    sc[top[0]] = self._in_scale_t if self._in_scale_t.device == img.device else self._in_scale_t.to(img.device)
            elif t in KIND:
                c = self.convs[str(i)]
                if _shapes_only:
                    y = c._conv(val[bot[0]], c.weight, c.bias)
                    if t == 74:
                        y = F.max_pool2d(y, 2, 2)
                    val[top[0]] = y
                    continue
                if act_on:
                    si_idx = self._conv_scale_idx.get(str(i))
                    s_out = scales_vec[si_idx] if si_idx is not None else None
                    val[top[0]] = c(val[bot[0]], sc[bot[0]], s_out=s_out)
                    sc[top[0]] = s_out
                else:
                    val[top[0]] = c(val[bot[0]], None)
            elif t == 33:
                for x in top:
                    val[x] = val[bot[0]]
                    if act_on:
                        sc[x] = sc.get(bot[0])
            elif t == 75:
                val[top[0]] = F.max_pool2d(val[bot[0]], 2, 2)
                if act_on:
                    sc[top[0]] = sc.get(bot[0])
            elif t == 69:
                v = val[bot[0]]
                val[top[0]] = zero_insert_up(v) if P[4] == 0 else F.interpolate(v, scale_factor=(P[2], P[3]), mode='nearest')
                if act_on:
                    sc[top[0]] = sc.get(bot[0])
            elif t == 71:
                val[top[0]] = torch.cat([val[b] for b in bot], 1)
                if act_on:
                    sc[top[0]] = torch.cat([sc[b].reshape(-1).expand(val[b].shape[1]) if sc[b].dim() == 0 or sc[b].numel() == 1
                                            else sc[b] for b in bot])
            elif t == 78:
                a, b = val[bot[0]], val[bot[1]]
                if _shapes_only or not act_on:
                    val[top[0]] = (a + b) / 2
                else:                                  # exact: mean of unsigned codes, half to even
                    s = sc[bot[0]]
                    if _HAS_TRITON and a.is_cuda and a.dtype == torch.float32 and P[0] <= 8:
                        val[top[0]] = _FusedType78.apply(a, b, s, P[0])
                    else:
                        # integer codes first: (a + b) / 2s in float can land at k + 0.4999999 instead of
                        # k + 0.5 and round the other way; the sum of two exact integers halved cannot
                        ua, ub = ste_round(a / s), ste_round(b / s)
                        u = ste_round((ua + ub) / 2).clamp(0, 2 ** P[0] - 1)
                        val[top[0]] = u * s
                if act_on:
                    sc[top[0]] = sc.get(bot[0])
            elif t == 76:
                return [val[b] for b in bot]
        raise ValueError('graph has no output layer')

    @torch.no_grad()
    def input_scales(self):
        """scale of each weight layer's input blob (for export), from the current scales."""
        sc, widths = {}, {}
        for i, (t, bot, top, P) in enumerate(self.spec):
            if t == 16:
                sc[top[0]] = torch.tensor(self.in_scale); widths[top[0]] = 3
            elif t in KIND:
                c = self.convs[str(i)]
                sc[top[0]] = c.act.scale.detach().cpu() if c.act is not None else None; widths[top[0]] = c.cout
            elif t == 33:
                for x in top:
                    sc[x], widths[x] = sc[bot[0]], widths[bot[0]]
            elif t in (69, 75, 78):
                sc[top[0]], widths[top[0]] = sc[bot[0]], widths[bot[0]]
            elif t == 71:
                sc[top[0]] = torch.cat([sc[b].reshape(-1).expand(widths[b]) if sc[b].numel() == 1 else sc[b] for b in bot])
                widths[top[0]] = sum(widths[b] for b in bot)
        return {i: sc[bot[0]] for i, (t, bot, top, P) in enumerate(self.spec) if t in KIND}


@torch.no_grad()
def from_jzdl(model, num_classes, in_h, in_w, class_map=None, seed=0, **kw):
    """GraphNet initialised from a JZDL model's own (dequantised) weights; activations in the
    model's LSB units (scale 1). class_map[k] = the model's class index for our class k, or None
    for a new class (small random weights, low-prior bias)."""
    g = torch.Generator().manual_seed(seed)
    jzdl_ref.VERSION[0] = model.version
    net = GraphNet(model, num_classes, in_h, in_w, **kw)
    for p in net.scales.values():
        p.fill_(0.0)
    for i, L in enumerate(model.layers):
        if L.type not in KIND:
            continue
        conv = net.convs[str(i)]
        if L.type == 46:
            d, _ = _dil(L.params)
            w = jzdl_ref.first_weights(L, conv.cin).transpose(0, 3, 1, 2)[:, :, ::d, ::d]
            bias, mult, sh = (jzdl_ref._i(L.blobs[k], np.int32) for k in (1, 2, 3))
            M = mult / 2.0 ** (31 + sh); z = 1 << (conv.bits_in - 1)
            s_in = net.in_scale if L.bottoms[0] == model.input_blob else 1.0
        elif L.type == 73:
            w = jzdl_ref.dw_weights(L, conv.cout).reshape(conv.cout, 1, 3, 3)
            bias, mult, sh = (jzdl_ref._i(L.blobs[k], np.int16) for k in (1, 2, 3))
            M = mult / 2.0 ** (15 + sh); z = 1 << (conv.bits_in - 1); s_in = 1.0
        elif L.type in (53, 74):
            w = jzdl_ref.pw_weights(L, conv.cin).reshape(conv.cout, conv.cin, 1, 1)
            bias, mult, sh = (jzdl_ref._i(L.blobs[k], np.int16) for k in (1, 2, 3))
            M = mult / 2.0 ** (15 + sh); z = 1 << (conv.bits_in - 1); s_in = 1.0
        else:
            co = L.params[0]
            w = jzdl_ref._i(L.blobs[0], np.int8).reshape(co, -1, 1, 1)
            bias = jzdl_ref._i(L.blobs[1], np.int32); M = np.frombuffer(L.blobs[2], np.float32).astype(np.float64)
            z = 1 << (conv.bits_in - 1); s_in = 1.0
        Wf = w * M.reshape(-1, 1, 1, 1) / s_in
        Bf = (bias - z * w.reshape(w.shape[0], -1).sum(1)) * M
        if L.type == 51:
            na = net.na
            old_nc = L.params[0] // na - 5
            per_old, per_new = 5 + old_nc, 5 + num_classes
            cmap = class_map if class_map is not None else list(range(num_classes))
            Wt, Bt = torch.from_numpy(Wf), torch.from_numpy(Bf)
            Wn = torch.zeros(na * per_new, *Wf.shape[1:], dtype=torch.float64); Bn = torch.zeros(na * per_new, dtype=torch.float64)
            for a in range(na):
                Wn[a * per_new:a * per_new + 5] = Wt[a * per_old:a * per_old + 5]
                Bn[a * per_new:a * per_new + 5] = Bt[a * per_old:a * per_old + 5]
                std = Wt[a * per_old + 5:a * per_old + per_old].std()
                for k in range(num_classes):
                    src = cmap[k]
                    if src is not None:
                        Wn[a * per_new + 5 + k] = Wt[a * per_old + 5 + src]; Bn[a * per_new + 5 + k] = Bt[a * per_old + 5 + src]
                    else:
                        Wn[a * per_new + 5 + k] = torch.randn(Wt.shape[1:], generator=g, dtype=torch.float64) * std * 0.1
                        Bn[a * per_new + 5 + k] = -4.6
            Wf, Bf = Wn.numpy(), Bn.numpy()
        conv.weight.data = torch.from_numpy(np.ascontiguousarray(Wf)).float()
        conv.bias.data = torch.from_numpy(np.ascontiguousarray(Bf)).float()
    return net


@torch.no_grad()
def export_graph(net, model, path=None):
    """Write net's weights into a copy of the JZDL model it was built from (same graph, version and
    layer order); the header takes the net's input size. Returns (jzdl.Model, accumulator bounds)."""
    from export import conv_ints, pack_first, pack_dw, pack_pw
    from calib import fit_requant
    net = net.eval().cpu()
    fit_requant(net)
    m = copy.deepcopy(model)
    m.in_h, m.in_w = net.in_h, net.in_w
    s_in = net.input_scales()
    v5 = m.version >= 5
    bounds = {}
    for i, L in enumerate(m.layers):
        if L.type not in KIND:
            continue
        conv = net.convs[str(i)]
        r = conv_ints(conv, s_in[i], conv.act.scale if conv.act is not None else None)
        bounds[i] = r['bound']
        P = list(L.params)
        P[0] = conv.cout
        if L.type == 46:
            w = r['w_int'].transpose(0, 2, 3, 1)                       # O, ky, kx, ci (dilation zero-stuffed)
            if v5 and w.shape[1:] != (3, 3, 3):
                raise ValueError('layer %d: v5 packing is only known for the 3x3 image conv; use a v4 graph' % i)
            wb = pack_first(w) if v5 else w.transpose(1, 2, 3, 0).astype(np.int8).reshape(-1)
            P[1], P[2] = w.shape[1], w.shape[2]; P[3] = P[4] = conv.conv_dil
            P[12] = len(wb); P[15] = conv.bits_in; P[16] = conv.bits_out
            L.blobs = [wb.tobytes(), r['bias'].astype(np.int32).tobytes(),
                       r['mult'].astype(np.int32).tobytes(), r['shift'].astype(np.int32).tobytes()]
        elif L.type == 73:
            w9 = r['w_int'].reshape(conv.cout, 9)
            wb = pack_dw(w9) if v5 else w9.T.astype(np.int8).reshape(-1)           # v4: [k][c]
            P[12] = len(wb); P[14] = conv.bits_in; P[15] = conv.bits_out
            L.blobs = [wb.tobytes()] + [a.astype(np.int16).tobytes() for a in (r['bias'], r['mult'], r['shift'], np.zeros(conv.cout))]
        elif L.type in (53, 74):
            w2 = r['w_int'].reshape(conv.cout, conv.cin)
            wb = pack_pw(w2) if v5 else w2.astype(np.int8).reshape(-1)              # v4: [out][in]
            P[12] = len(wb); P[14] = conv.bits_in; P[15] = conv.bits_out
            L.blobs = [wb.tobytes()] + [a.astype(np.int16).tobytes() for a in (r['bias'], r['mult'], r['shift'], np.zeros(conv.cout))]
        else:
            wb = r['w_int'].reshape(conv.cout, conv.cin).astype(np.int8)
            P[12] = wb.size; P[13] = conv.bits_in
            L.blobs = [wb.tobytes(), r['bias'].astype(np.int32).tobytes(), r['q'].astype(np.float32).tobytes()]
        if L.type in (73, 53, 74) and np.abs(r['bias']).max() > 32767:
            raise ValueError('layer %d: int16 bias overflow' % i)
        L.params = P
    data = m.to_bytes()
    if path:
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            f.write(data)
        os.replace(tmp, path)
    return m, bounds
