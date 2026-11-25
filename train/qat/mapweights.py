"""Map float pretrained weights onto a graphgen-built GraphNet, exactly.

Every blob carries a per-channel (gain g, offset o): true value = g * stored - o.
  - gain: libjzdl's residual (type 78) stores the MEAN of its inputs, so its output has gain 2g.
    A residual's two inputs must share one gain: the branch's last conv takes the skip's gain
    (node extra match=<skip blob>).
  - offset: T31 activations are unsigned. A layer without activation in the source network
    (node extra linear=True: MobileNet's dw_start / pw_proj) stores true + o. o comes from
    offsets[blob] (measured: -(channel minimum on calibration images) with a margin) or, without
    data, max(0, k_sigma * |gamma| - beta) from its BatchNorm (a poor guess once the graph differs
    from the source: measure when you can). Values below -o are clipped. A residual's offset is
    the sum of its inputs' offsets.
A consumer folds its input's (g, o) into its weights and bias: W' = W * g, b' = b - sum(W * o).
Zero padding is stored 0 (true -o): exact everywhere except at the borders of a depthwise conv
whose input carries an offset.

Weights come from sd with keys '<src>.conv.weight' and '<src>.bn.{weight,bias,running_mean,running_var}'.
A node with src None keeps GraphNet's random weights and gets a fresh BatchNorm (new layers,
e.g. adapters); with extra identity=True it starts as the identity (a re-centring depthwise layer:
centre tap 1), which resets its input's (gain, offset) to (1, its own offset). A full conv whose source kernel is larger than the graph's (5x5 -> 3x3 depthwise)
takes the centre taps.
"""
import torch
import torch.nn as nn


@torch.no_grad()
def map_weights(net, sd, first_fn=None, head_fn=None, bn_eps=1e-5, k_sigma=4.0, offsets=None):
    """first_fn(W) -> (W, bias) for the image conv (input conventions); head_fn(conv, extra) fills a
    head (type 51) in true-input units. Returns {blob name: (gain, offset)}."""
    convs = iter(i for i, s in enumerate(net.spec) if s[0] in (46, 73, 53, 51))
    go = {'image': (torch.ones(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64))}
    for t, ins, out, P, src, ex in net.graph_nodes:
        if t == 78:
            (ga, oa), (gb, ob) = go[ins[0]], go[ins[1]]
            if not torch.allclose(ga, gb):
                raise AssertionError('residual %s: gains differ' % out)
            go[out] = (2 * ga, oa + ob)
            continue
        if t == 71:
            go[out] = (torch.cat([go[x][0] for x in ins]), torch.cat([go[x][1] for x in ins]))
            continue
        if t == 69:
            go[out] = go[ins[0]]
            continue
        c = net.convs[str(next(convs))]
        gi, oi = go[ins[0]]
        if t == 51:
            head_fn(c, ex)
            W, b = c.weight.double(), c.bias.double()
            b = b - W[:, :, 0, 0] @ oi
            c.weight.copy_((W * gi.view(1, -1, 1, 1)).float()); c.bias.copy_(b.float())
            go[out] = None
            continue
        if ex.get('identity'):                                   # re-centring layer: centre tap 1, no BatchNorm effect
            W = torch.zeros_like(c.weight, dtype=torch.float64)
            W[:, :, W.shape[2] // 2, W.shape[3] // 2] = 1.0
            bias = torch.zeros(W.shape[0], dtype=torch.float64)
            gamma, beta = torch.ones(W.shape[0], dtype=torch.float64), torch.zeros(W.shape[0], dtype=torch.float64)
            mean, var = torch.zeros(W.shape[0]), torch.full((W.shape[0],), 1.0 - bn_eps)
        elif src is None:
            W = c.weight.double()
            bias = torch.zeros(W.shape[0], dtype=torch.float64)
            gamma, beta = torch.ones(W.shape[0], dtype=torch.float64), torch.zeros(W.shape[0], dtype=torch.float64)
            mean, var = torch.zeros(W.shape[0]), torch.ones(W.shape[0])
        else:
            W = sd[src + '.conv.weight'].double()
            bias = torch.zeros(W.shape[0], dtype=torch.float64)
            gamma, beta = sd[src + '.bn.weight'].double(), sd[src + '.bn.bias'].double()
            mean, var = sd[src + '.bn.running_mean'], sd[src + '.bn.running_var']
            if t == 46 and ins[0] == 'image' and first_fn is not None:
                W, bias = first_fn(W)
            if ex.get('spp'):
                W = W.view(W.shape[0], ex['spp'], -1, 1, 1).sum(1)
            k = c.weight.shape[-1]
            if W.shape[-1] > k:                                  # e.g. 5x5 depthwise -> 3x3 on the T31
                a = (W.shape[-1] - k) // 2
                W = W[:, :, a:a + k, a:a + k]
        if t == 73:
            bias = bias - W.sum((1, 2, 3)) * oi
            W = W * gi.view(-1, 1, 1, 1)
        else:
            bias = bias - (W.sum((2, 3)) * oi.view(1, -1)).sum(1)
            W = W * gi.view(1, -1, 1, 1)
        g_out = go[ex['match']][0] if ex.get('match') else torch.ones(W.shape[0], dtype=torch.float64)
        if not ex.get('linear'):
            o_out = torch.zeros(W.shape[0], dtype=torch.float64)
        elif offsets is not None and out in offsets:
            o_out = offsets[out].double()
        else:
            o_out = (k_sigma * gamma.abs() - beta).clamp_min(0)
        bnm = nn.BatchNorm2d(W.shape[0], eps=bn_eps)
        bnm.weight.copy_((gamma / g_out).float()); bnm.bias.copy_(((beta + o_out) / g_out).float())
        bnm.running_mean.copy_(mean); bnm.running_var.copy_(var)
        if c.weight.shape != W.shape:
            raise ValueError('%s: weight %s != graph %s' % (src, tuple(W.shape), tuple(c.weight.shape)))
        c.weight.copy_(W.float()); c.bias.copy_(bias.float())
        c.bn, c.bn_folded = bnm, False
        go[out] = (g_out, o_out)
    for c in net.convs.values():
        c.quant_w = False
        if c.act is not None:
            c.act.enabled = False
    net.blob_go = go
    return go
