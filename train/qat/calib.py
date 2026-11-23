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


@torch.no_grad()
def calibrate_acts(net, batches, pct=0.9999):
    """Set each ActQ scale so the pct quantile of its (ReLU'd) input maps to the top code."""
    acts, hooks = {}, []
    for name, m in net.L.items():
        if m.act is not None:
            m.act.enabled = False
            hooks.append(m.act.register_forward_hook(
                lambda mod, inp, out, name=name: acts.setdefault(name, []).append(torch.relu(inp[0]).flatten())))
    net.eval()
    for b in batches:
        net(b)
    for h in hooks:
        h.remove()
    for name, vals in acts.items():
        v = torch.cat(vals)
        v = v[torch.randperm(v.numel())[:2_000_000]]
        top = torch.quantile(v, pct).clamp_min(1e-6)
        a = net.L[name].act
        a.log_s.copy_(torch.log(top / (2 ** a.bits - 1)))
        a.enabled = True
    for m in net.modules():
        if isinstance(m, QConv):
            m.quant_w = True
