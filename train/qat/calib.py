"""BatchNorm folding and activation-scale calibration for qat.net.Detector."""
import torch
from net import QConv


@torch.no_grad()
def bn_stats(net, batches):
    """Collect BatchNorm running stats with quantisation off (cumulative average)."""
    for m in net.modules():
        if isinstance(m, QConv):
            m.quant_w = False
            if m.act is not None:
                m.act.enabled = False
            if m.bn is not None:
                m.bn.reset_running_stats(); m.bn.momentum = None
    net.train()
    for b in batches:
        net(b)
    net.eval()


@torch.no_grad()
def fold_bn(net):
    for m in net.modules():
        if isinstance(m, QConv) and m.bn is not None and not m.bn_folded:
            W, B = m.folded()
            m.weight.copy_(W); m.bias.copy_(B); m.bn_folded = True


def _sample(t, n=250_000):
    """A random subset of a tensor's values (keeps calibration memory small: a first-layer
    activation of one 64-image batch alone is > 100M values)."""
    t = t.flatten()
    return t if t.numel() <= n else t[torch.randint(t.numel(), (n,), device=t.device)].cpu()


@torch.no_grad()
def calibrate_acts(net, batches, pct=0.9999):
    """Set each ActQ scale so the pct quantile of its (ReLU'd) input maps to the top code.
    Works on a Detector (net.L) or a GraphNet (net.convs); layers that share one scale (GraphNet's
    tied blobs) get the largest of their ranges."""
    acts, hooks = {}, []
    layers = net.L if hasattr(net, 'L') else net.convs
    for name, m in layers.items():
        if m.act is not None:
            m.act.enabled = False
            hooks.append(m.act.register_forward_hook(
                lambda mod, inp, out, name=name: acts.setdefault(name, []).append(_sample(torch.relu(inp[0])))))
    net.eval()
    for b in batches:
        net(b)
    for h in hooks:
        h.remove()
    tops = {}
    for name, vals in acts.items():
        v = torch.cat(vals)
        v = v[torch.randperm(v.numel())[:2_000_000]]
        a = layers[name].act
        top = (torch.quantile(v, pct).clamp_min(1e-6) / (2 ** a.bits - 1)).item()
        tops[id(a.log_s)] = max(tops.get(id(a.log_s), 0.0), top)
    for name in acts:
        a = layers[name].act
        a.log_s.copy_(torch.log(torch.tensor(tops[id(a.log_s)])))
        a.enabled = True
    for m in net.modules():
        if isinstance(m, QConv):
            m.quant_w = True


@torch.no_grad()
def fit_requant(net, rounds=4):
    """GraphNet: raise any activation scale that breaks libjzdl's requant limit (each output
    channel's weight step q = max|W * s_in| / wmax must stay below the output scale, or the shift
    goes negative and the layer can't be exported). New, randomly initialised layers can start
    there. Raising s_out only costs that tensor some resolution. Scales tied across blobs move
    together; later layers' inputs change, hence a few rounds. Returns the layers raised."""
    from export import WMAX
    from net import quant_params, MAX_SH
    if not hasattr(net, 'spec'):
        return []
    raised = []
    for _ in range(rounds):
        s_in = net.input_scales()
        changed = False
        for i, c in net.convs.items():
            if c.act is None:
                continue
            W, B = c.folded()
            Wp = (W * c.in_scale(s_in[int(i)].float())).float()
            q, m, sh = quant_params(Wp, B.float(), c.bits_in, WMAX[c.kind if c.kind != 'pwpool' else 'pw'],
                                    c.act.scale.float(), getattr(c, 'acc_signed', False), MAX_SH[c.kind])
            if (sh < 0).any():
                c.act.log_s.copy_(torch.log(q.max() * 1.01))
                raised.append(int(i))
                changed = True
        if not changed:
            break
    return sorted(set(raised))
