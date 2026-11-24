"""T41 export: the trained Detector as a plain float ONNX for Ingenic's NNA converter.

One model, two exports (see docs/one-model-two-exports.md):
  T31  export.py      -> JZDL .bin, integer-exact (the QAT forward IS the camera's arithmetic)
  T41  export_t41.py  -> float ONNX -> magik-transform-tools (post-training int8) -> venus .bin

The ONNX is written op by op (no torch.onnx exporter), with only ops that Ingenic's converter
(magik-transform-tools 2.0.4) was tested to accept: Conv, Relu, MaxPool, ConvTranspose, Concat.
  - BatchNorm is already folded; weights are the same folded W, B the T31 export starts from.
  - Activations keep the T31 range clamp(y, 0, t), t = (2^bits - 1) * s, without the rounding.
    The clamp is essential (without it AP50 drops to ~0: the model relies on those ceilings), but
    the converter rejects Clip in every form (opset 10 attributes, opset 11+ inputs). So it is built
    from accepted ops, exactly:  min(relu(y), t) = t - relu(t - relu(y))
      Relu -> depthwise 1x1 Conv (w = -1, b = t) -> Relu -> depthwise 1x1 Conv (w = -1, b = t)
  - persondet's zero-insertion upsample = depthwise ConvTranspose 2x2 stride 2 with kernel
    [[1, 0], [0, 0]] (exact). Resize + Mul by a 0/1 mask is also exact but the converter fails
    on the Mul (MAGIK_CODE_NOT_MATCHED_DATA).
Input "images": 1x3xHxW RGB in 0..1 (the cfg's NORMAL 255 maps camera pixels to that).
Outputs "h32", "h16": raw head logits NCHW, 3 x (5 + classes) channels, decoded exactly as on T31.
"""
import json, os
import numpy as np
import torch
import torch.nn.functional as F

OPSET = 12
POOL = dict(ceil_mode=0, kernel_shape=[2, 2], pads=[0, 0, 0, 0], strides=[2, 2])
CHAIN = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9']      # up to the stride-16 branch


def _params(net):
    """Folded float weights + activation clip max per conv, from a trained Detector."""
    out = {}
    with torch.no_grad():
        for name, conv in net.L.items():
            W, B = conv.folded()
            top = None
            if conv.act is not None:
                top = float(conv.act.scale) * ((1 << conv.bits_out) - 1)
            out[name] = (W.detach().float().cpu().numpy(), B.detach().float().cpu().numpy(), top, conv.kind)
    return out


class FloatDetector(torch.nn.Module):
    """PyTorch twin of the exported ONNX (for accuracy checks). forward(img 0..255).
    clip=False drops the upper clamp (for comparison only: AP collapses)."""

    def __init__(self, net, clip=True):
        super().__init__()
        self.p = _params(net)
        self.clip = clip
        self._dev = {}

    def _conv(self, name, x):
        W, B, top, kind = self.p[name]
        key = (name, str(x.device))
        if key not in self._dev:
            self._dev[key] = (torch.from_numpy(W).to(x.device), torch.from_numpy(B).to(x.device))
        W, B = self._dev[key]
        if kind == 'first':
            y = F.conv2d(F.pad(x, (0, 1, 0, 1)), W, B, stride=2)
        elif kind == 'dw':
            y = F.conv2d(F.pad(x, (1, 1, 1, 1)), W, B, groups=W.shape[0])
        else:
            y = F.conv2d(x, W, B)
        if top is None:
            return y
        y = y.clamp(0, top) if self.clip else F.relu(y)
        return F.max_pool2d(y, 2, 2) if kind == 'pwpool' else y

    def forward(self, img):
        x = img / 255.0
        for n in CHAIN:
            x = self._conv(n, x)
        b9 = x
        x = F.max_pool2d(b9, 2, 2)
        for n in ('l12', 'l13', 'l14', 'l15', 'l16'):
            x = self._conv(n, x)
        b17 = x
        h32 = self._conv('l20', self._conv('l19', self._conv('l18', b17)))
        z = self._conv('l21', b17)
        u = torch.zeros(z.shape[0], z.shape[1], z.shape[2] * 2, z.shape[3] * 2, device=z.device)
        u[:, :, ::2, ::2] = z
        z = torch.cat([u, b9], 1)
        h16 = self._conv('l26', self._conv('l25', self._conv('l24', z)))
        return h32, h16


def build_onnx(net, in_h, in_w):
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    p = _params(net)
    nodes, inits = [], []
    const = lambda name, arr: inits.append(numpy_helper.from_array(np.asarray(arr), name)) or name

    def conv(name, x):
        W, B, top, kind = p[name]
        const(name + '.w', W.astype(np.float32)); const(name + '.b', B.astype(np.float32))
        # every attribute explicit: the converter does not apply ONNX defaults
        attrs = dict(dilations=[1, 1], group=1, kernel_shape=list(W.shape[2:]), pads=[0, 0, 0, 0], strides=[1, 1])
        if kind == 'first':
            attrs.update(strides=[2, 2], pads=[0, 0, 1, 1])          # top, left, bottom, right
        elif kind == 'dw':
            attrs.update(pads=[1, 1, 1, 1], group=int(W.shape[0]))
        y = name + ('.conv' if top is not None else '')
        if top is None:
            y = {'l20': 'h32', 'l26': 'h16'}[name]
        nodes.append(helper.make_node('Conv', [x, name + '.w', name + '.b'], [y], name=name, **attrs))
        if top is None:
            return y
        # clamp(y, 0, t) = t - relu(t - relu(y)), from Relu + depthwise 1x1 convs (no Clip)
        C = int(W.shape[0])
        neg = const(name + '.neg_w', np.full((C, 1, 1, 1), -1, np.float32))
        tb = const(name + '.t', np.full(C, top, np.float32))
        dw1 = dict(dilations=[1, 1], group=C, kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1])
        nodes.append(helper.make_node('Relu', [y], [name + '.r1'], name=name + '_relu1'))
        nodes.append(helper.make_node('Conv', [name + '.r1', neg, tb], [name + '.d1'], name=name + '_tminus1', **dw1))
        nodes.append(helper.make_node('Relu', [name + '.d1'], [name + '.r2'], name=name + '_relu2'))
        a = name + '.act'
        nodes.append(helper.make_node('Conv', [name + '.r2', neg, tb], [a], name=name + '_tminus2', **dw1))
        if kind == 'pwpool':
            nodes.append(helper.make_node('MaxPool', [a], [name + '.pool'], name=name + '_pool', **POOL))
            return name + '.pool'
        return a

    x = 'images'
    for n in CHAIN:
        x = conv(n, x)
    b9 = x
    nodes.append(helper.make_node('MaxPool', [b9], ['p10'], name='pool10', **POOL))
    x = 'p10'
    for n in ('l12', 'l13', 'l14', 'l15', 'l16'):
        x = conv(n, x)
    b17 = x
    conv('l20', conv('l19', conv('l18', b17)))
    z = conv('l21', b17)
    # zero-insertion x2: depthwise ConvTranspose, kernel [[1, 0], [0, 0]], stride 2
    c = p['l21'][0].shape[0]
    k = np.zeros((c, 1, 2, 2), np.float32); k[:, 0, 0, 0] = 1
    nodes.append(helper.make_node('ConvTranspose', [z, const('up.w', k)], ['up'], name='up_deconv', dilations=[1, 1],
                                  group=int(c), kernel_shape=[2, 2], pads=[0, 0, 0, 0], strides=[2, 2],
                                  output_padding=[0, 0]))
    nodes.append(helper.make_node('Concat', ['up', b9], ['cat'], name='cat', axis=1))
    conv('l26', conv('l25', conv('l24', 'cat')))
    no = p['l20'][0].shape[0]
    graph = helper.make_graph(
        nodes, 'thingino_ai_detector',
        [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, in_h, in_w])],
        [helper.make_tensor_value_info('h32', TensorProto.FLOAT, [1, no, in_h // 32, in_w // 32]),
         helper.make_tensor_value_info('h16', TensorProto.FLOAT, [1, no, in_h // 16, in_w // 16])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', OPSET)], producer_name='thingino-ai')
    model.ir_version = 7                                               # readable by older onnx (converter: 1.12)
    onnx.checker.check_model(model)
    return model


def magik_cfg(in_h, in_w, calib_dir='calib', outputs=('h32', 'h16')):
    return {'SOC': 'T41', 'INPUT': ['images'], 'INPUT_SHAPE': [1, 3, in_h, in_w], 'MEAN': [0, 0, 0],
            'NORMAL': [255.0, 255.0, 255.0], 'COLOR': ['RGB'], 'OUTPUT': list(outputs),
            'QUANT_DATASET_PATH': './%s/' % calib_dir}


def export_t41(net, in_h, in_w, out_dir, calib_frames=None, check_frames=None):
    """Write out_dir/{model_t41.onnx, magik_t41.cfg, calib/*.jpg}. Returns a report dict.
    calib_frames: list of HxWx3 uint8 RGB already at in_h x in_w (the converter's calibration set).
    check_frames: frames for the ONNX == FloatDetector check (onnxruntime, if installed)."""
    import onnx
    os.makedirs(out_dir, exist_ok=True)
    net = net.eval().cpu()
    graph = hasattr(net, 'spec')
    model = build_onnx_graph(net, in_h, in_w) if graph else build_onnx(net, in_h, in_w)
    path = os.path.join(out_dir, 'model_t41.onnx')
    onnx.save(model, path)
    json.dump(magik_cfg(in_h, in_w, outputs=head_names(net)), open(os.path.join(out_dir, 'magik_t41.cfg'), 'w'), indent=1)
    rep = dict(onnx_bytes=os.path.getsize(path), opset=OPSET)
    if calib_frames is not None:
        from PIL import Image
        cd = os.path.join(out_dir, 'calib')
        os.makedirs(cd, exist_ok=True)
        for i, f in enumerate(calib_frames):
            Image.fromarray(f).save(os.path.join(cd, '%04d.jpg' % i), quality=95)
        rep['calib_images'] = len(calib_frames)
    if check_frames:
        fd = FloatGraph(net) if graph else FloatDetector(net)
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        except ImportError:
            sess = None
        worst = 0.0
        for f in check_frames:
            x = torch.from_numpy(f).permute(2, 0, 1)[None].float()
            with torch.no_grad():
                tor = fd(x)
            if sess is not None:
                ort_out = sess.run(head_names(net), {'images': (x / 255.0).numpy()})
                worst = max([worst] + [float(np.abs(o - tt.numpy()).max()) for o, tt in zip(ort_out, tor)])
        rep['onnx_vs_torch_max_diff'] = worst if sess is not None else None
    return rep


# ------------------------------------------------------------------ any JZDL graph (graphnet.GraphNet)
def head_names(net):
    return ['h%d' % s for s in net.strides] if hasattr(net, 'strides') else ['h32', 'h16']


class FloatGraph(torch.nn.Module):
    """Float twin of a GraphNet's ONNX export: folded weights, clamp(y, 0, top) activations, mean
    for residuals, no rounding. forward(img 0..255) -> head logits in output order."""

    def __init__(self, net, clip=True):
        super().__init__()
        self.net, self.clip = net, clip
        self.strides, self.anchors = net.strides, net.anchors

    @torch.no_grad()
    def forward(self, img):
        import torch.nn.functional as F
        from graphnet import KIND
        n = self.net
        val = {}
        for i, (t, bot, top, P) in enumerate(n.spec):
            if t == 16:
                val[top[0]] = img / 255.0
            elif t in KIND:
                c = n.convs[str(i)]
                W, B = c.folded()
                y = c._conv(val[bot[0]], W.to(img.device), B.to(img.device))
                if c.act is not None:
                    tp = float(c.act.scale) * ((1 << c.bits_out) - 1)
                    y = y.clamp(0, tp) if self.clip else F.relu(y)
                    if t == 74:
                        y = F.max_pool2d(y, 2, 2)
                val[top[0]] = y
            elif t == 33:
                for x in top:
                    val[x] = val[bot[0]]
            elif t == 75:
                val[top[0]] = F.max_pool2d(val[bot[0]], 2, 2)
            elif t == 69:
                v = val[bot[0]]
                if P[4] == 0:
                    u = torch.zeros(v.shape[0], v.shape[1], v.shape[2] * 2, v.shape[3] * 2, device=v.device)
                    u[:, :, ::2, ::2] = v
                    val[top[0]] = u
                else:
                    val[top[0]] = F.interpolate(v, scale_factor=2, mode='nearest')
            elif t == 71:
                val[top[0]] = torch.cat([val[b] for b in bot], 1)
            elif t == 78:
                val[top[0]] = (val[bot[0]] + val[bot[1]]) / 2
            elif t == 76:
                return [val[b] for b in bot]


def build_onnx_graph(net, in_h, in_w, clip=True):
    """ONNX of a GraphNet with only converter-accepted ops (see build_onnx for the persondet rules)."""
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    from graphnet import KIND
    nodes, inits = [], []
    const = lambda name, arr: inits.append(numpy_helper.from_array(np.asarray(arr), name)) or name
    names = {}
    dw1 = lambda C: dict(dilations=[1, 1], group=C, kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1])
    outs = []
    for i, (t, bot, top, P) in enumerate(net.spec):
        tag = 'n%d' % i
        if t == 16:
            names[top[0]] = 'images'
        elif t in KIND:
            c = net.convs[str(i)]
            W, B = c.folded()
            W = W.detach().float().cpu().numpy(); B = B.detach().float().cpu().numpy()
            l, r, tt, bb = c.conv_pad
            attrs = dict(dilations=[1, 1], group=int(W.shape[0]) if c.kind == 'dw' else 1,
                         kernel_shape=list(W.shape[2:]), pads=[tt, l, bb, r], strides=[c.conv_stride] * 2)
            y = tag + '.conv'
            if c.act is None:
                y = 'h%d' % net.strides[net.head_layers.index(i)]; outs.append(y)
            nodes.append(helper.make_node('Conv', [names[bot[0]], const(tag + '.w', W), const(tag + '.b', B)], [y], name=tag, **attrs))
            if c.act is not None and not clip:             # experiment: plain ReLU, no upper clamp
                nodes.append(helper.make_node('Relu', [y], [tag + '.act'], name=tag + '_relu'))
                y = tag + '.act'
                if t == 74:
                    nodes.append(helper.make_node('MaxPool', [y], [tag + '.pool'], name=tag + '_pool', **POOL)); y = tag + '.pool'
                names[top[0]] = y
                continue
            if c.act is not None:
                C = int(W.shape[0]); tp = float(c.act.scale.detach()) * ((1 << c.bits_out) - 1)
                neg = const(tag + '.neg', np.full((C, 1, 1, 1), -1, np.float32)); tb = const(tag + '.t', np.full(C, tp, np.float32))
                nodes.append(helper.make_node('Relu', [y], [tag + '.r1'], name=tag + '_r1'))
                nodes.append(helper.make_node('Conv', [tag + '.r1', neg, tb], [tag + '.d1'], name=tag + '_m1', **dw1(C)))
                nodes.append(helper.make_node('Relu', [tag + '.d1'], [tag + '.r2'], name=tag + '_r2'))
                nodes.append(helper.make_node('Conv', [tag + '.r2', neg, tb], [tag + '.act'], name=tag + '_m2', **dw1(C)))
                y = tag + '.act'
                if t == 74:
                    nodes.append(helper.make_node('MaxPool', [y], [tag + '.pool'], name=tag + '_pool', **POOL)); y = tag + '.pool'
            names[top[0]] = y
        elif t == 33:
            for x in top:
                names[x] = names[bot[0]]
        elif t == 75:
            nodes.append(helper.make_node('MaxPool', [names[bot[0]]], [tag], name=tag, **POOL)); names[top[0]] = tag
        elif t == 69:
            C = _width(net, bot[0])
            k = np.zeros((C, 1, 2, 2), np.float32)
            if P[4] == 0:
                k[:, 0, 0, 0] = 1                      # zero insertion
            else:
                k[:] = 1                               # nearest neighbour: each value fills its 2x2 block
            nodes.append(helper.make_node('ConvTranspose', [names[bot[0]], const(tag + '.k', k)], [tag], name=tag, dilations=[1, 1],
                                          group=C, kernel_shape=[2, 2], pads=[0, 0, 0, 0], strides=[2, 2], output_padding=[0, 0]))
            names[top[0]] = tag
        elif t == 71:
            nodes.append(helper.make_node('Concat', [names[b] for b in bot], [tag], name=tag, axis=1)); names[top[0]] = tag
        elif t == 78:
            C = _width(net, bot[0])
            nodes.append(helper.make_node('Add', [names[bot[0]], names[bot[1]]], [tag + '.sum'], name=tag + '_add'))
            nodes.append(helper.make_node('Conv', [tag + '.sum', const(tag + '.half', np.full((C, 1, 1, 1), 0.5, np.float32)),
                                                   const(tag + '.z', np.zeros(C, np.float32))], [tag], name=tag + '_half', **dw1(C)))
            names[top[0]] = tag
    no = net.no
    graph = helper.make_graph(nodes, 'thingino_ai_detector',
                              [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, in_h, in_w])],
                              [helper.make_tensor_value_info('h%d' % s, TensorProto.FLOAT, [1, no, in_h // s, in_w // s])
                               for s in net.strides], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', OPSET)], producer_name='thingino-ai')
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _width(net, blob):
    """channel count of a blob in a GraphNet (from a shapes-only forward, cached)."""
    if not hasattr(net, '_widths'):
        import torch.nn.functional as F
        from graphnet import KIND
        w = {}
        for i, (t, bot, top, P) in enumerate(net.spec):
            if t == 16:
                w[top[0]] = 3
            elif t in KIND:
                w[top[0]] = net.convs[str(i)].cout
            elif t == 33:
                for x in top:
                    w[x] = w[bot[0]]
            elif t in (69, 75, 78):
                w[top[0]] = w[bot[0]]
            elif t == 71:
                w[top[0]] = sum(w[b] for b in bot)
        net._widths = w
    return net._widths[blob]
