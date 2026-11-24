"""Export a trained qat.net.Detector to a libjzdl version-5 .bin (persondet graph).

Integer mapping per conv (see docs/jzdl-format.md "Arithmetic"):
  W' = W * s_in (real per input LSB), q = weight_step(W', B)   (int16 accumulator bound)
  w_int = round(W' / q)
  bias_hw = round(B / q) + Z_in * sum(w_int)     # hardware convolves signed x = u - Z_in
  M = q / s_out -> mult, shift with M = mult / 2^(Q + shift)   (Q = 15, first conv 31)
  head: bias int32 = bias_hw, scale float32 = q   ->  logits = (acc + bias) * scale
"""
import copy, math, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'tools', 'jzdl'))
import jzdl  # noqa: E402
from net import WMAX, Detector, quant_params  # noqa: E402

# Ingenic's persondet (public: gtxaspec/magik_models, Txx/jzdl/). Never committed here:
# build_model.py downloads it into train/cache/. Override with JZDL_TEMPLATE.
DEFAULT_TEMPLATE = os.path.join(HERE, '..', 'cache', 'magik_model_persondet.bin')


def template_path():
    p = os.environ.get('JZDL_TEMPLATE') or DEFAULT_TEMPLATE
    if not os.path.exists(p):
        raise SystemExit('persondet template not found: %s (run build_model.py, or set JZDL_TEMPLATE)' % p)
    return p

LAYER_INDEX = {'l1': 1, 'l2': 2, 'l3': 3, 'l4': 4, 'l5': 5, 'l6': 6, 'l7': 7, 'l8': 8, 'l9': 9,
               'l12': 12, 'l13': 13, 'l14': 14, 'l15': 15, 'l16': 16, 'l18': 18, 'l19': 19, 'l20': 20,
               'l21': 21, 'l24': 24, 'l25': 25, 'l26': 26}
DW_POS = [0, 1, 2, 3, 4, 5, 6, 7, 9]


def mult_shift(M, q_bits):
    """M (array, 0 < M < 1) -> (mult, shift) with M ~= mult / 2^(q_bits + shift), mult in [2^(q-1), 2^q)."""
    M = np.asarray(M, np.float64)
    if np.any(M >= 1) or np.any(M <= 0):
        raise ValueError('requant scale out of range (need 0 < M < 1): min %g max %g' % (M.min(), M.max()))
    sh = np.floor(-np.log2(M)).astype(np.int64)
    m = np.round(M * 2.0 ** (q_bits + sh)).astype(np.int64)
    over = m >= (1 << q_bits)
    m[over] >>= 1
    sh[over] -= 1
    if np.any(sh < 0):
        raise ValueError('negative shift')
    return m, sh


def pack_first(w):            # w [O, 3, 3, Cin]  (o, ky, kx, ci)
    O, _, _, cin = w.shape
    per = 30 * 16 if cin == 3 else None
    out = np.zeros((O // 16) * per, np.int8)
    for g in range(O // 16):
        for ky in range(3):
            for kx in range(3):
                for ci in range(cin):
                    s = ky * 10 + kx * cin + ci
                    for o in range(16):
                        out[g * per + (s // 2) * 32 + o * 2 + s % 2] = w[g * 16 + o, ky, kx, ci]
    return out


def pack_dw(w):               # w [C, 9]
    C = w.shape[0]
    out = np.zeros(C * 10, np.int8)
    for g in range(C // 16):
        for t in range(9):
            p = DW_POS[t]
            for o in range(16):
                out[g * 160 + (p // 2) * 32 + o * 2 + p % 2] = w[g * 16 + o, t]
    return out


def pack_pw(w):               # w [O, Cin]
    O, cin = w.shape
    ncp = cin // 2
    out = np.zeros(O * cin, np.int8)
    for og in range(O // 16):
        for cp in range(ncp):
            blk = w[og * 16:og * 16 + 16, cp * 2:cp * 2 + 2]
            out[(og * ncp + cp) * 32:(og * ncp + cp + 1) * 32] = blk.reshape(-1)
    return out


@torch.no_grad()
def conv_ints(conv, s_in, s_out):
    """Integer parameters, computed with the same quant_params() the training forward uses."""
    W, B = conv.folded()
    si = conv.in_scale(s_in.float())
    Wp = (W * si).float(); B = B.float()
    wmax = WMAX[conv.kind if conv.kind != 'pwpool' else 'pw']
    q, m, sh = quant_params(Wp, B, conv.bits_in, wmax, None if s_out is None else s_out.float())
    w_int = torch.round(Wp / q.view(-1, 1, 1, 1)).to(torch.int64)
    b_int = torch.round(B / q).to(torch.int64)
    umax = (1 << conv.bits_in) - 1
    wf = w_int.flatten(1)
    ch = torch.maximum((b_int + umax * wf.clamp(min=0).sum(1)).abs(), (b_int + umax * wf.clamp(max=0).sum(1)).abs())
    bound = ch.max().item()
    if bound > 32767:
        raise ValueError('%s: int16 accumulator bound %d > 32767' % (conv.kind, bound))
    z_in = 1 << (conv.bits_in - 1)
    r = dict(w_int=w_int.numpy(), bias=(b_int + z_in * wf.sum(1)).numpy(), q=q.numpy(), bound=bound)
    if conv.kind != 'head':
        if (sh < 0).any():
            raise ValueError('%s: requant scale >= 1 (q > s_out): output scale too small' % conv.kind)
        m = m.to(torch.int64).numpy(); sh = sh.to(torch.int64).numpy()
        r['mult'] = (m << 16) if conv.kind == 'first' else m      # first conv: Q31 = Q15 << 16
        r['shift'] = sh
    return r


@torch.no_grad()
def export(net, in_h, in_w, path=None):
    net = net.eval().cpu()
    from skeleton import skeleton
    m = skeleton(net, in_h, in_w)                      # graph from the network definition, no template
    Ls = net.L
    sc = lambda n: Ls[n].act.scale
    s_img = torch.tensor(net.in_scale)
    s_cat = torch.cat([sc('l21').expand(net.C['c128']), sc('l9').expand(net.C['c256'])])
    inputs = {'l1': s_img, 'l2': sc('l1'), 'l3': sc('l2'), 'l4': sc('l3'), 'l5': sc('l4'), 'l6': sc('l5'),
              'l7': sc('l6'), 'l8': sc('l7'), 'l9': sc('l8'), 'l12': sc('l9'), 'l13': sc('l12'),
              'l14': sc('l13'), 'l15': sc('l14'), 'l16': sc('l15'), 'l18': sc('l16'), 'l19': sc('l18'),
              'l20': sc('l19'), 'l21': sc('l16'), 'l24': s_cat, 'l25': sc('l24'), 'l26': sc('l25')}
    report = {}
    for name, li in LAYER_INDEX.items():
        conv = Ls[name]
        L = m.layers[li]
        s_out = conv.act.scale if conv.act is not None else None
        r = conv_ints(conv, inputs[name], s_out)
        report[name] = r['bound']
        P = list(L.params)
        P[0] = conv.cout
        if conv.kind == 'first':
            w = r['w_int'].transpose(0, 2, 3, 1)                 # O, ky, kx, ci
            wb = pack_first(w)
            P[12] = len(wb); P[16] = conv.bits_out
            L.blobs = [wb.tobytes(), r['bias'].astype(np.int32).tobytes(),
                       r['mult'].astype(np.int32).tobytes(), r['shift'].astype(np.int32).tobytes()]
        elif conv.kind == 'dw':
            wb = pack_dw(r['w_int'].reshape(conv.cout, 9))
            P[12] = len(wb); P[14] = conv.bits_in; P[15] = conv.bits_out
            L.blobs = [wb.tobytes()] + [a.astype(np.int16).tobytes() for a in
                                        (r['bias'], r['mult'], r['shift'], np.zeros(conv.cout))]
        elif conv.kind in ('pw', 'pwpool'):
            wb = pack_pw(r['w_int'].reshape(conv.cout, conv.cin))
            P[12] = len(wb); P[14] = conv.bits_in; P[15] = conv.bits_out
            L.blobs = [wb.tobytes()] + [a.astype(np.int16).tobytes() for a in
                                        (r['bias'], r['mult'], r['shift'], np.zeros(conv.cout))]
        else:  # head
            wb = r['w_int'].reshape(conv.cout, conv.cin).astype(np.int8)
            P[12] = wb.size; P[13] = conv.bits_in
            L.blobs = [wb.tobytes(), r['bias'].astype(np.int32).tobytes(), r['q'].astype(np.float32).tobytes()]
        for a in (r['bias'],) if conv.kind in ('dw', 'pw', 'pwpool') else ():
            if np.abs(a).max() > 32767:
                raise ValueError('%s: int16 bias overflow' % name)
        L.params = P
    data = m.to_bytes()
    if path:
        open(path, 'wb').write(data)
    return m, report
