"""Initialise a qat.net.Detector from Ingenic's persondet (dequantised), heads widened to 3 x (5 + N).

Activations are kept in persondet's own units (scale 1 = one activation LSB), so
the float net starts out computing persondet's function:
  W = w_int * M / s_in,  B = (bias_hw - Z_in * sum(w_int)) * M
  M = mult / 2^(Q + shift) for hidden convs, M = scale for the heads.
Head rows per anchor: persondet (x, y, w, h, obj, person) -> ours (x, y, w, h, obj, person, cls2, ...).
New class rows start with small weights and a low-prior bias (sigmoid ~ 0.01).
BatchNorm is not used (bn_folded = True): we fine-tune from a working network.
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..', 'tools', 'jzdl'))
import jzdl, jzdl_ref  # noqa: E402
from jzdl_ref import _i, first_weights, dw_weights, pw_weights  # noqa: E402
from net import Detector  # noqa: E402
from export import LAYER_INDEX, template_path  # noqa: E402


@torch.no_grad()
def from_persondet(num_classes=3, path=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    pd = jzdl.Model.parse(open(path or template_path(), 'rb').read())
    net = Detector(num_classes=num_classes)
    for name, li in LAYER_INDEX.items():
        conv, L = net.L[name], pd.layers[li]
        conv.bn_folded = True
        conv.bn = None
        if conv.act is not None:
            conv.act.log_s.fill_(0.0)                       # scale 1 = persondet LSB
        s_in = net.in_scale if conv.kind == 'first' else 1.0
        if conv.kind == 'first':
            w = first_weights(L, 3).transpose(0, 3, 1, 2)    # O, ci, ky, kx
            bias, mult, sh = _i(L.blobs[1], np.int32), _i(L.blobs[2], np.int32), _i(L.blobs[3], np.int32)
            M = mult / 2.0 ** (31 + sh); z = 128
        elif conv.kind == 'dw':
            w = dw_weights(L, conv.cout).reshape(conv.cout, 1, 3, 3)
            bias, mult, sh = _i(L.blobs[1], np.int16), _i(L.blobs[2], np.int16), _i(L.blobs[3], np.int16)
            M = mult / 2.0 ** (15 + sh); z = 1 << (L.params[14] - 1)
        elif conv.kind in ('pw', 'pwpool'):
            w = pw_weights(L, conv.cin).reshape(conv.cout, conv.cin, 1, 1)
            bias, mult, sh = _i(L.blobs[1], np.int16), _i(L.blobs[2], np.int16), _i(L.blobs[3], np.int16)
            M = mult / 2.0 ** (15 + sh); z = 1 << (L.params[14] - 1)
        else:   # head: expand 3 x 6 -> 3 x (5 + N)
            co = L.params[0]
            w = _i(L.blobs[0], np.int8).reshape(co, -1, 1, 1)
            bias = _i(L.blobs[1], np.int32); M = np.frombuffer(L.blobs[2], np.float32).astype(np.float64)
            z = 1 << (L.params[13] - 1)
        Wf = w * M.reshape(-1, 1, 1, 1) / s_in
        Bf = (bias - z * w.reshape(w.shape[0], -1).sum(1)) * M
        if conv.kind == 'head':
            per_old, per_new = 6, 5 + num_classes
            Wn = torch.zeros(3 * per_new, *Wf.shape[1:], dtype=torch.float64)
            Bn = torch.zeros(3 * per_new, dtype=torch.float64)
            Wt, Bt = torch.from_numpy(Wf), torch.from_numpy(Bf)
            for a in range(3):
                Wn[a * per_new:a * per_new + 6] = Wt[a * per_old:a * per_old + 6]
                Bn[a * per_new:a * per_new + 6] = Bt[a * per_old:a * per_old + 6]
                std = Wt[a * per_old + 5].std()
                for k in range(6, per_new):
                    Wn[a * per_new + k] = torch.randn(Wt.shape[1:], generator=g, dtype=torch.float64) * std * 0.1
                    Bn[a * per_new + k] = -4.6
            Wf, Bf = Wn.numpy(), Bn.numpy()
        conv.weight.data = torch.from_numpy(np.ascontiguousarray(Wf)).float()
        conv.bias.data = torch.from_numpy(np.ascontiguousarray(Bf)).float()
    return net

