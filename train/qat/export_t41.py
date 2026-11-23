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


def magik_cfg(in_h, in_w, calib_dir='calib'):
    return {'SOC': 'T41', 'INPUT': ['images'], 'INPUT_SHAPE': [1, 3, in_h, in_w], 'MEAN': [0, 0, 0],
            'NORMAL': [255.0, 255.0, 255.0], 'COLOR': ['RGB'], 'OUTPUT': ['h32', 'h16'],
            'QUANT_DATASET_PATH': './%s/' % calib_dir}


def export_t41(net, in_h, in_w, out_dir, calib_frames=None, check_frames=None):
    """Write out_dir/{model_t41.onnx, magik_t41.cfg, calib/*.jpg}. Returns a report dict.
    calib_frames: list of HxWx3 uint8 RGB already at in_h x in_w (the converter's calibration set).
    check_frames: frames for the ONNX == FloatDetector check (onnxruntime, if installed)."""
    import onnx
    os.makedirs(out_dir, exist_ok=True)
    net = net.eval().cpu()
    model = build_onnx(net, in_h, in_w)
    path = os.path.join(out_dir, 'model_t41.onnx')
    onnx.save(model, path)
    json.dump(magik_cfg(in_h, in_w), open(os.path.join(out_dir, 'magik_t41.cfg'), 'w'), indent=1)
    rep = dict(onnx_bytes=os.path.getsize(path), opset=OPSET)
    if calib_frames is not None:
        from PIL import Image
        cd = os.path.join(out_dir, 'calib')
        os.makedirs(cd, exist_ok=True)
        for i, f in enumerate(calib_frames):
            Image.fromarray(f).save(os.path.join(cd, '%04d.jpg' % i), quality=95)
        rep['calib_images'] = len(calib_frames)
    if check_frames:
        fd = FloatDetector(net)
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        except ImportError:
            sess = None
        worst = 0.0
        for f in check_frames:
            x = torch.from_numpy(f).permute(2, 0, 1)[None].float()
            with torch.no_grad():
                t32, t16 = fd(x)
            if sess is not None:
                o32, o16 = sess.run(['h32', 'h16'], {'images': (x / 255.0).numpy()})
                worst = max(worst, float(np.abs(o32 - t32.numpy()).max()), float(np.abs(o16 - t16.numpy()).max()))
        rep['onnx_vs_torch_max_diff'] = worst if sess is not None else None
    return rep
