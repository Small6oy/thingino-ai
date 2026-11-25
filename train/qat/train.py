#!/usr/bin/env python3
"""QAT training of the person / pet / package detector (hardware-exact forward).

  train.py DATA_ROOT [--epochs 30] [--bs 16] [--lr 5e-4] [--out runs/x] [--limit N]
                     [--workers N] [--device cuda|xpu|mps|cpu] [--pkg-weight 4.0]

Normally run through ../build_model.py. Re-running with the same --out resumes from
last.pt (model, optimiser and schedule).

DATA_ROOT/{train,val}/{images,labels} from data_prep.py. Starts from Ingenic's persondet
(init_persondet.py), heads 3 x (5 + 3). Geometry matches raptor/aitest: a random 16:9
crop stretched to 416x320, RGB. Decode/anchors match aitest headed_decode (YOLOv5 style,
yolov3-tiny anchors). Each epoch: eval mAP@0.5 per class and export a .bin.
"""
import contextlib, glob, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def _is_cuda(dev):
    return str(dev).startswith('cuda')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from init_persondet import from_persondet  # noqa: E402
from export import export, repair_export_scales  # noqa: E402

H, W = 320, 416
# THINGINO_AI_INPUT (build_model.py --input WxH, yoloxn / mnv4n only): the model's input size. Both
# multiples of 32 (three stride-2 levels below 8). 384x224 and 448x256 keep the camera's 16:9.
_IN = os.environ.get('THINGINO_AI_INPUT', '').strip()
if _IN:
    try:
        W, H = (int(v) for v in _IN.lower().split('x'))
    except ValueError:
        raise SystemExit('input %r: want WxH, e.g. 384x224' % _IN)
    if W % 32 or H % 32 or W < 64 or H < 64:
        raise SystemExit('input %dx%d: width and height must be multiples of 32' % (W, H))
# class list: train/manifest/classes.json (written by data/build_manifest.py); v1 had 3 classes
_CJ = os.path.join(HERE, '..', 'manifest', 'classes.json')
LABEL_CLASSES = json.load(open(_CJ))['classes'] if os.path.exists(_CJ) else ['person', 'pet', 'package']   # ids in the label files
# THINGINO_AI_CLASSES (build_model.py --classes person,pet,vehicle): train a subset of the manifest's
# classes, in that order. Boxes of the other classes are dropped (their images stay, as background).
_SEL = [c.strip() for c in os.environ.get('THINGINO_AI_CLASSES', '').split(',') if c.strip()]
for _c in _SEL:
    if _c not in LABEL_CLASSES:
        raise SystemExit('unknown class %r (the manifest has: %s)' % (_c, ', '.join(LABEL_CLASSES)))
CLASSES = _SEL or list(LABEL_CLASSES)
NC = len(CLASSES)
LABEL_MAP = {i: CLASSES.index(c) for i, c in enumerate(LABEL_CLASSES) if c in CLASSES}   # label id -> model class
PACKAGE = CLASSES.index('package') if 'package' in CLASSES else None
ANCHORS = {32: [(81, 82), (135, 169), (344, 319)], 16: [(10, 14), (23, 27), (37, 58)]}
BALANCE = {8: 4.0, 16: 1.0, 32: 0.4}                 # objectness weight per stride (YOLOv5's P3/P4/P5)


def head_dict(net, outs):
    """Net outputs -> {stride: logits}. Detector (persondet) returns (h32, h16); a GraphNet
    returns its heads in output order with net.strides."""
    strides = getattr(net, 'strides', None) or [32, 16]
    return dict(zip(strides, outs))


THRESH = {'person': 40, 'vehicle': 40}     # suggested per-class thresholds (%); others 30 (live tests)


def model_conf(net, classes=None, in_h=None, in_w=None):
    """The model's sidecar, model.conf: rvd's [motion] keys for this model, so a camera runs it
    without hand-typed settings (raptor.conf's own [motion] keys override these). INI text."""
    classes = list(classes or CLASSES)
    anc = anchors_of(net)
    strides = sorted(int(s) for s in anc)
    lines = ['# thingino-ai model sidecar: rvd [motion] defaults for this model',
             '# (keys in raptor.conf [motion] override these)',
             '[motion]',
             'algorithm = yolo',
             'num_classes = %d' % len(classes),
             'class_names = %s' % ','.join(classes),
             'class_thresholds = %s' % ','.join(str(THRESH.get(c, 30)) for c in classes)]
    for st in strides:
        lines.append('anchors_%d = %s' % (st, ','.join('%g,%g' % tuple(a) for a in anc[st])))
    lines.append('input = %dx%d' % (in_w or W, in_h or H))
    lines.append('preprocess = stretch')
    return '\n'.join(lines) + '\n'


def anchors_of(net):
    """A GraphNet carries its anchors as {stride: [...]}. persondet's Detector has an int
    'anchors' (anchors per cell, 3), which must not be mistaken for that."""
    a = getattr(net, 'anchors', None)
    return a if isinstance(a, dict) else ANCHORS


# ------------------------------------------------------------------ data
def label_path(img):
    """split/images/x.jpg -> split/labels/x.txt (works with either path separator)."""
    d, f = os.path.split(img)
    return os.path.join(os.path.dirname(d), 'labels', os.path.splitext(f)[0] + '.txt')


def _load_labels(lp):
    """Fast plain-Python YOLO label parser (matches np.loadtxt(..., ndmin=2).reshape(-1, 5)). Class ids
    are mapped from the manifest's list to CLASSES; boxes of classes not trained are dropped."""
    rows = []
    with open(lp, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 5:
                c = LABEL_MAP.get(int(float(parts[0])))
                if c is not None:
                    rows.append((float(c), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5) if rows else np.zeros((0, 5), dtype=np.float64)


class DetSet(Dataset):
    def __init__(self, root, split, train, limit=None, gpu_aug=False):
        idx = os.path.join(root, split, 'index.txt')   # written by build_model.py: this build's images only
        if os.path.exists(idx):
            with open(idx) as f:
                self.imgs = sorted(os.path.join(root, split, 'images', n.strip() + '.jpg') for n in f if n.strip())
        else:
            self.imgs = sorted(glob.glob(os.path.join(root, split, 'images', '*.jpg')))
        if limit:
            random.Random(0).shuffle(self.imgs); self.imgs = self.imgs[:limit]
        self.labels = [_load_labels(label_path(p)) for p in self.imgs]
        self.train = train
        self.gpu_aug = gpu_aug and train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        from PIL import Image, ImageEnhance
        p = self.imgs[i]
        with Image.open(p) as _im:
            im = _im.convert('RGB')
        lab = self.labels[i]
        iw, ih = im.size
        # boxes to pixel x0 y0 x1 y1
        b = np.stack([lab[:, 1] - lab[:, 3] / 2, lab[:, 2] - lab[:, 4] / 2,
                      lab[:, 1] + lab[:, 3] / 2, lab[:, 2] + lab[:, 4] / 2], 1) * [iw, ih, iw, ih] if len(lab) else np.zeros((0, 4))
        cls = lab[:, 0].astype(np.int64)
        # 16:9 crop (random scale/position when training, centred otherwise)
        if iw / ih > 16 / 9:
            cw, ch = ih * 16 / 9, ih
        else:
            cw, ch = iw, iw * 9 / 16
        if self.train:
            s = random.uniform(0.6, 1.0); cw, ch = cw * s, ch * s
            x0 = random.uniform(0, iw - cw); y0 = random.uniform(0, ih - ch)
        else:
            x0, y0 = (iw - cw) / 2, (ih - ch) / 2
        im = im.crop((round(x0), round(y0), round(x0 + cw), round(y0 + ch))).resize((W, H), Image.BILINEAR)
        if len(b):
            area0 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            b = b - [x0, y0, x0, y0]
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, cw); b[:, [1, 3]] = b[:, [1, 3]].clip(0, ch)
            area1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            keep = (area1 > 0.35 * area0) & (b[:, 2] - b[:, 0] > 2) & (b[:, 3] - b[:, 1] > 2)
            b, cls = b[keep] * [W / cw, H / ch, W / cw, H / ch], cls[keep]
        if self.train:
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                if len(b):
                    b[:, [0, 2]] = W - b[:, [2, 0]]
            if not self.gpu_aug:
                im = ImageEnhance.Brightness(im).enhance(random.uniform(0.6, 1.4))
                im = ImageEnhance.Contrast(im).enhance(random.uniform(0.7, 1.3))
                im = ImageEnhance.Color(im).enhance(random.uniform(0.5, 1.5))
                if random.random() < 0.25:                       # IR night mode: grayscale
                    im = im.convert('L').convert('RGB')
        arr = torch.from_numpy(np.asarray(im, np.uint8).copy()).permute(2, 0, 1)
        x = arr if self.gpu_aug else arr.float()
        t = torch.zeros((len(b), 6))
        if len(b):
            t[:, 1] = torch.from_numpy(cls).float()
            t[:, 2:] = torch.from_numpy(np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                                                  b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)).float()
        return x, t


if _HAS_TRITON:
    @triton.jit
    def _gpu_aug_pass1_luma(x_ptr, fb_ptr, luma_sum_ptr, HW, BLOCK: tl.constexpr):
        n = tl.program_id(0)
        pid_hw = tl.program_id(1)
        offs = pid_hw * BLOCK + tl.arange(0, BLOCK)
        mask = offs < HW
        base = n * 3 * HW + offs
        fb = tl.load(fb_ptr + n)
        r = tl.minimum(tl.load(x_ptr + base, mask=mask, other=0).to(tl.float32) * fb, 255.0)
        g = tl.minimum(tl.load(x_ptr + base + HW, mask=mask, other=0).to(tl.float32) * fb, 255.0)
        b = tl.minimum(tl.load(x_ptr + base + 2 * HW, mask=mask, other=0).to(tl.float32) * fb, 255.0)
        luma = tl.where(mask, r * 0.299 + g * 0.587 + b * 0.114, 0.0)
        tl.atomic_add(luma_sum_ptr + n, tl.sum(luma, axis=0))

    @triton.jit
    def _gpu_aug_pass2_apply(
        x_ptr, fb_ptr, fc_ptr, fs_ptr, ir_ptr, mean_ptr, out_ptr,
        HW, OUT_NHWC: tl.constexpr, BLOCK: tl.constexpr,
    ):
        n = tl.program_id(0)
        pid_hw = tl.program_id(1)
        offs = pid_hw * BLOCK + tl.arange(0, BLOCK)
        mask = offs < HW
        base = n * 3 * HW + offs
        fb = tl.load(fb_ptr + n)
        fc = tl.load(fc_ptr + n)
        fs = tl.load(fs_ptr + n)
        ir = tl.load(ir_ptr + n) != 0
        mean = tl.load(mean_ptr + n)
        r = tl.minimum(tl.load(x_ptr + base, mask=mask, other=0).to(tl.float32) * fb, 255.0)
        g = tl.minimum(tl.load(x_ptr + base + HW, mask=mask, other=0).to(tl.float32) * fb, 255.0)
        b = tl.minimum(tl.load(x_ptr + base + 2 * HW, mask=mask, other=0).to(tl.float32) * fb, 255.0)
        r = tl.minimum(tl.maximum(mean + (r - mean) * fc, 0.0), 255.0)
        g = tl.minimum(tl.maximum(mean + (g - mean) * fc, 0.0), 255.0)
        b = tl.minimum(tl.maximum(mean + (b - mean) * fc, 0.0), 255.0)
        gray = r * 0.299 + g * 0.587 + b * 0.114
        r = tl.minimum(tl.maximum(gray + (r - gray) * fs, 0.0), 255.0)
        g = tl.minimum(tl.maximum(gray + (g - gray) * fs, 0.0), 255.0)
        b = tl.minimum(tl.maximum(gray + (b - gray) * fs, 0.0), 255.0)
        gray2 = r * 0.299 + g * 0.587 + b * 0.114
        r = tl.where(ir, gray2, r)
        g = tl.where(ir, gray2, g)
        b = tl.where(ir, gray2, b)
        r = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(r), 0.0), 255.0)
        g = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(g), 0.0), 255.0)
        b = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(b), 0.0), 255.0)
        if OUT_NHWC:
            ob = n * 3 * HW + offs * 3
            tl.store(out_ptr + ob, r, mask=mask)
            tl.store(out_ptr + ob + 1, g, mask=mask)
            tl.store(out_ptr + ob + 2, b, mask=mask)
        else:
            tl.store(out_ptr + base, r, mask=mask)
            tl.store(out_ptr + base + HW, g, mask=mask)
            tl.store(out_ptr + base + 2 * HW, b, mask=mask)


@torch.no_grad()
def _gpu_color_aug(x, channels_last=False):
    """Vectorized GPU equivalent of PIL Brightness/Contrast/Color/Grayscale jitter.
    Returns float32 tensor with exact integer 0..255 values (matching uint8 PIL output)."""
    if x.dtype != torch.uint8:
        return x.to(memory_format=torch.channels_last) if channels_last else x
    if _HAS_TRITON and x.is_cuda and x.dim() == 4 and x.shape[1] == 3 and x.numel() > 0:
        x = x.contiguous()
        n, _, h, w = x.shape
        hw = h * w
        dev = x.device
        fb = torch.empty((n,), device=dev).uniform_(0.6, 1.4)
        fc = torch.empty((n,), device=dev).uniform_(0.7, 1.3)
        fs = torch.empty((n,), device=dev).uniform_(0.5, 1.5)
        ir = (torch.rand((n,), device=dev) < 0.25).to(torch.uint8)
        luma_sum = torch.zeros((n,), device=dev, dtype=torch.float32)
        grid = (n, (hw + 1023) // 1024)
        _gpu_aug_pass1_luma[grid](x, fb, luma_sum, hw, BLOCK=1024)
        mean = luma_sum / float(hw)
        mf = torch.channels_last if channels_last else torch.contiguous_format
        out = torch.empty((n, 3, h, w), device=dev, dtype=torch.float32, memory_format=mf)
        _gpu_aug_pass2_apply[grid](x, fb, fc, fs, ir, mean, out, hw, OUT_NHWC=channels_last, BLOCK=1024)
        return out
    x = x.float()
    n = x.shape[0]
    dev = x.device
    # Brightness: blend with black (0.6..1.4)
    fb = torch.empty((n, 1, 1, 1), device=dev).uniform_(0.6, 1.4)
    x = (x * fb).clamp_(0.0, 255.0)
    # Contrast: blend with per-image mean luminance (0.7..1.3)
    luma = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
    mean = luma.mean(dim=(2, 3), keepdim=True)
    fc = torch.empty((n, 1, 1, 1), device=dev).uniform_(0.7, 1.3)
    x = (mean + (x - mean) * fc).clamp_(0.0, 255.0)
    # Color: blend with grayscale (0.5..1.5)
    gray = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
    fs = torch.empty((n, 1, 1, 1), device=dev).uniform_(0.5, 1.5)
    x = (gray + (x - gray) * fs).clamp_(0.0, 255.0)
    # IR night mode: 25% grayscale (unconditional torch.where avoids GPU->CPU sync)
    ir = torch.rand((n, 1, 1, 1), device=dev) < 0.25
    gray2 = (x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114).expand_as(x)
    x = torch.where(ir, gray2, x)
    x = x.round_().clamp_(0.0, 255.0)
    return x.to(memory_format=torch.channels_last) if channels_last else x


def collate(batch):
    xs, ts = zip(*batch)
    for i, t in enumerate(ts):
        t[:, 0] = i
    return torch.stack(xs), torch.cat(ts)


# ------------------------------------------------------------------ loss (YOLOv5 style)
def bbox_ciou(b1, b2, eps=1e-7):
    """b1, b2: (n,4) cx cy w h"""
    b1x1, b1x2 = b1[:, 0] - b1[:, 2] / 2, b1[:, 0] + b1[:, 2] / 2
    b1y1, b1y2 = b1[:, 1] - b1[:, 3] / 2, b1[:, 1] + b1[:, 3] / 2
    b2x1, b2x2 = b2[:, 0] - b2[:, 2] / 2, b2[:, 0] + b2[:, 2] / 2
    b2y1, b2y2 = b2[:, 1] - b2[:, 3] / 2, b2[:, 1] + b2[:, 3] / 2
    inter = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(0) * (torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)).clamp(0)
    union = b1[:, 2] * b1[:, 3] + b2[:, 2] * b2[:, 3] - inter + eps
    iou = inter / union
    cw = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1); ch = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = (b1[:, 0] - b2[:, 0]) ** 2 + (b1[:, 1] - b2[:, 1]) ** 2
    v = (4 / math.pi ** 2) * (torch.atan(b2[:, 2] / (b2[:, 3] + eps)) - torch.atan(b1[:, 2] / (b1[:, 3] + eps))) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


_LOSS_CONST_CACHE = {}
_DECODE_GRID_CACHE = {}


def _loss_consts(dev, s, gh, gw, anchors=None):
    anchors = anchors or ANCHORS
    key = (dev, s, gh, gw, tuple(anchors[s]))
    c = _LOSS_CONST_CACHE.get(key)
    if c is None:
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=dev, dtype=torch.float32) * 0.5
        anc = torch.tensor(anchors[s], device=dev, dtype=torch.float32) / s
        gwh = torch.tensor([gw, gh], device=dev, dtype=torch.float32)
        c = (off, anc, gwh)
        _LOSS_CONST_CACHE[key] = c
    return c


def _decode_consts(dev, s, gh, gw, anchors=None):
    anchors = anchors or ANCHORS
    key = (dev, s, gh, gw, tuple(anchors[s]))
    c = _DECODE_GRID_CACHE.get(key)
    if c is None:
        anc = torch.tensor(anchors[s], device=dev, dtype=torch.float32)
        gy, gx = torch.meshgrid(torch.arange(gh, device=dev), torch.arange(gw, device=dev), indexing='ij')
        c = (anc, gy, gx)
        _DECODE_GRID_CACHE[key] = c
    return c


def yolo_loss(outs, targets, nc=NC, anchors=None):
    """outs: {stride: (N, 3*(5+nc), gh, gw)} ; targets (n, 6): img, cls, cx, cy, w, h (pixels)"""
    dev = targets.device
    if len(targets):          # never index with a bad target (seen once on XPU: 'index out of bounds' at epoch 26)
        ok = (torch.isfinite(targets).all(1) & (targets[:, 1] >= 0) & (targets[:, 1] < nc)
              & (targets[:, 4] > 0) & (targets[:, 5] > 0))
        targets = targets[ok]
        if MIN_BOX_TRAIN[0] > 0:            # too small to resolve: leave them out of the loss (--min-box)
            targets = targets[(targets[:, 4] * targets[:, 5]).sqrt() >= MIN_BOX_TRAIN[0]]
    lbox = torch.zeros(1, device=dev); lobj = torch.zeros(1, device=dev); lcls = torch.zeros(1, device=dev)
    bce = torch.nn.BCEWithLogitsLoss()
    for s, p in outs.items():
        n, _, gh, gw = p.shape
        off, anc, gwh = _loss_consts(dev, s, gh, gw, anchors)
        p = p.view(n, 3, 5 + nc, gh, gw).permute(0, 1, 3, 4, 2)       # n, a, y, x, k
        tobj = torch.zeros(p.shape[:4], device=dev)
        if len(targets):
            t = targets.clone(); t[:, 2:] /= s                          # grid units
            na = 3
            ta = t.unsqueeze(0).repeat(na, 1, 1)                        # a, nt, 6
            ai = torch.arange(na, device=dev).view(na, 1).repeat(1, len(t))
            r = ta[..., 4:6] / anc[:, None]
            j = torch.max(r, 1 / r).max(2).values < 4.0
            ta, ai = ta[j], ai[j]
            if not len(ta):                 # no target fits this stride's anchors: empty mean would be NaN
                lobj += bce(p[..., 4], tobj) * BALANCE[s]
                continue
            gxy = ta[:, 2:4]; gxi = gwh - gxy
            jj, kk = ((gxy % 1 < 0.5) & (gxy > 1)).T
            ll, mm = ((gxi % 1 < 0.5) & (gxi > 1)).T
            sel = torch.stack((torch.ones_like(jj), jj, kk, ll, mm))
            ta = ta.repeat((5, 1, 1))[sel]; ai = ai.repeat((5, 1))[sel]
            offs = (torch.zeros_like(gxy)[None] + off[:, None])[sel]
            gij = (ta[:, 2:4] - offs).long()
            gi = gij[:, 0].clamp(0, gw - 1); gj = gij[:, 1].clamp(0, gh - 1)
            b = ta[:, 0].long().clamp(0, n - 1)
            ps = p[b, ai, gj, gi]
            pxy = ps[:, :2].sigmoid() * 2 - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anc[ai]
            tbox = torch.cat((ta[:, 2:4] - torch.stack([gi, gj], 1).float(), ta[:, 4:6]), 1)
            iou = bbox_ciou(torch.cat((pxy, pwh), 1), tbox)
            lbox += (1 - iou).mean()
            tobj[b, ai, gj, gi] = iou.detach().clamp(0).to(tobj.dtype)
            tc = torch.zeros((len(ps), nc), device=dev); tc[torch.arange(len(ps), device=dev), ta[:, 1].long()] = 1
            lcls += bce(ps[:, 5:], tc)
        lobj += bce(p[..., 4], tobj) * BALANCE[s]
    return lbox * 0.05 + lobj * 1.0 + lcls * 0.5, torch.cat((lbox, lobj, lcls)).detach()


# ------------------------------------------------------------------ decode + eval (as aitest)
def decode(outs, conf=0.05, nc=NC, anchors=None):
    """-> per image list of (x0, y0, x1, y1, score, cls), aitest headed_decode semantics"""
    from torchvision.ops import nms
    res = None
    for s, p in outs.items():
        n, _, gh, gw = p.shape
        p = p.view(n, 3, 5 + nc, gh, gw).permute(0, 1, 3, 4, 2).sigmoid()
        anc, gy, gx = _decode_consts(p.device, s, gh, gw, anchors)
        cx = (p[..., 0] * 2 - 0.5 + gx) * W / gw; cy = (p[..., 1] * 2 - 0.5 + gy) * H / gh
        bw = (p[..., 2] * 2) ** 2 * anc[:, 0].view(1, 3, 1, 1); bh = (p[..., 3] * 2) ** 2 * anc[:, 1].view(1, 3, 1, 1)
        sc, c = p[..., 5:].max(-1)
        score = p[..., 4] * sc
        d = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, score, c.float()], -1).view(n, -1, 6)
        res = d if res is None else torch.cat([res, d], 1)
    res = res.cpu()
    out = []
    for d in res:
        d = d[d[:, 4] > conf]
        keep = []
        for c in range(nc):
            dc = d[d[:, 5] == c]
            if len(dc):
                k = nms(dc[:, :4], dc[:, 4], 0.45)
                keep.append(dc[k[:100]])
        out.append(torch.cat(keep) if keep else torch.zeros((0, 6)))
    return out


def ap50(dets, gts, nc=NC):
    """dets/gts: lists per image; gts (cls, cx, cy, w, h). Returns AP@0.5 per class."""
    from torchvision.ops import box_iou
    aps = []
    for c in range(nc):
        recs, npos = [], 0
        for d, g in zip(dets, gts):
            g = g[g[:, 0] == c]; npos += len(g)
            d = d[d[:, 5] == c]
            if not len(d):
                continue
            d = d[d[:, 4].argsort(descending=True)]
            used = torch.zeros(len(g), dtype=torch.bool)
            gb = torch.stack([g[:, 1] - g[:, 3] / 2, g[:, 2] - g[:, 4] / 2, g[:, 1] + g[:, 3] / 2, g[:, 2] + g[:, 4] / 2], 1) if len(g) else None
            ious = box_iou(d[:, :4].cpu(), gb.cpu()) if len(g) else None
            for k in range(len(d)):
                tp = 0
                if len(g):
                    v, j = ious[k].max(0)
                    if v >= 0.5 and not used[j]:
                        used[j] = True; tp = 1
                recs.append((d[k, 4].item(), tp))
        if npos == 0:
            aps.append(float('nan')); continue
        if not recs:
            aps.append(0.0); continue
        recs.sort(key=lambda r: -r[0])
        tp = np.cumsum([r[1] for r in recs]); fp = np.cumsum([1 - r[1] for r in recs])
        rec = tp / npos; prec = tp / np.maximum(tp + fp, 1e-9)
        env = np.flip(np.maximum.accumulate(np.flip(prec)))     # precision envelope
        # COCO-style 101-point AP: precision at recall >= x, 0 beyond the max recall reached
        x = np.linspace(0, 1, 101)
        idx = np.searchsorted(rec, x, side='left')
        aps.append(float(np.mean(np.where(idx < len(rec), env[np.minimum(idx, len(rec) - 1)], 0.0))))
    return aps


MIN_BOX = 12          # px at the model input: smaller boxes are ignored in training (--min-box) and in the sized APs
SIZE_BUCKETS = {'small': (12, 32), 'medium_large': (32, 1e9)}      # sqrt(w*h) in px at 416x320


def ap50_sized(dets, gts, lo, hi, nc=NC):
    """AP@0.5 per class over objects of size sqrt(w*h) in [lo, hi) px, COCO-style: ground truth
    outside the range is 'ignore' (a detection on it is neither TP nor FP), and an unmatched
    detection outside the range is ignored too."""
    from torchvision.ops import box_iou
    aps = []
    for c in range(nc):
        recs, npos = [], 0
        for d, g in zip(dets, gts):
            g = g[g[:, 0] == c]; d = d[d[:, 5] == c]
            gsz = (g[:, 3] * g[:, 4]).clamp(min=0).sqrt() if len(g) else torch.zeros(0)
            inr = (gsz >= lo) & (gsz < hi)
            npos += int(inr.sum())
            if not len(d):
                continue
            d = d[d[:, 4].argsort(descending=True)]
            dsz = ((d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])).clamp(min=0).sqrt()
            used = torch.zeros(len(g), dtype=torch.bool)
            ious = None
            if len(g):
                gb = torch.stack([g[:, 1] - g[:, 3] / 2, g[:, 2] - g[:, 4] / 2, g[:, 1] + g[:, 3] / 2, g[:, 2] + g[:, 4] / 2], 1)
                ious = box_iou(d[:, :4].cpu(), gb.cpu())
            for k in range(len(d)):
                m = -1
                if ious is not None:                                  # best unused in-range gt, else best ignore gt
                    iu = ious[k].clone(); iu[used] = -1
                    cand = torch.where(inr, iu, torch.full_like(iu, -1))
                    v, j = cand.max(0)
                    if v >= 0.5:
                        m = int(j)
                    else:
                        v2, j2 = torch.where(~inr, ious[k], torch.full_like(iu, -1)).max(0)
                        if v2 >= 0.5:
                            continue                                  # on an ignored object
                if m >= 0:
                    used[m] = True; recs.append((d[k, 4].item(), 1))
                elif lo <= float(dsz[k]) < hi:
                    recs.append((d[k, 4].item(), 0))
        if npos == 0:
            aps.append(float('nan')); continue
        if not recs:
            aps.append(0.0); continue
        recs.sort(key=lambda r: -r[0])
        tp = np.cumsum([r[1] for r in recs]); fp = np.cumsum([1 - r[1] for r in recs])
        rec = tp / npos; prec = tp / np.maximum(tp + fp, 1e-9)
        env = np.flip(np.maximum.accumulate(np.flip(prec)))
        x = np.linspace(0, 1, 101)
        idx = np.searchsorted(rec, x, side='left')
        aps.append(float(np.mean(np.where(idx < len(rec), env[np.minimum(idx, len(rec) - 1)], 0.0))))
    return aps


LAST_EVAL = {}        # evaluate() also leaves the sized APs here: {'small': {...}, 'medium_large': {...}}
MIN_BOX_TRAIN = [MIN_BOX]   # set by build_model --min-box; 0 = train on every box (v1/v2 behaviour)


@torch.no_grad()
def evaluate(net, loader, dev, amp_bf16=False, channels_last=False):
    net.eval()
    dets, gts = [], []
    amp_ctx = (lambda: torch.amp.autocast('cuda', dtype=torch.bfloat16)) if (amp_bf16 and _is_cuda(dev)) else contextlib.nullcontext
    for x, t in loader:
        x = x.to(dev, memory_format=torch.channels_last if channels_last else torch.preserve_format, non_blocking=True)
        with amp_ctx():
            raw = net(x)
        outs = {s: o.float() for s, o in head_dict(net, raw).items()}
        dets += decode(outs, anchors=anchors_of(net))
        for i in range(len(x)):
            gts.append(t[t[:, 0] == i][:, 1:])
    LAST_EVAL.clear()
    for k, (lo, hi) in SIZE_BUCKETS.items():
        LAST_EVAL[k] = dict(zip(CLASSES, ap50_sized(dets, gts, lo, hi)))
    return ap50(dets, gts)


# ------------------------------------------------------------------ main
def pick_device(want=None):
    if want:
        return want
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def setup_exact_math():
    """The fake-quant forward relies on exact float32 integer arithmetic: no TF32."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        xb = getattr(torch.backends, 'xpu', None)
        if xb is not None and hasattr(xb, 'matmul'):
            xb.matmul.allow_tf32 = False


# ------------------------------------------------------------------ architectures
ARCHS = ('persondet', 'yolon')
# Ingenic's YOLOv5 sample classes: person, bicycle, car, motorcycle -> ours (by name)
YOLON_SRC_CLASSES = ['person', 'bicycle', 'car', 'motorcycle']
YOLON_FROM = {'person': 'person', 'vehicle': 'car'}


def yolon_template_path():
    p = os.environ.get('JZDL_YOLON_TEMPLATE') or os.path.join(HERE, '..', 'cache', 'magik_model_yolov5.bin')
    if not os.path.exists(p):
        raise SystemExit('YOLOv5 template not found: %s (run build_model.py --arch yolon, or set JZDL_YOLON_TEMPLATE)' % p)
    return p


ARCHS = {'persondet': ('persondet', 'scratch'), 'yolon': ('persondet',), 'yoloxn': ('yolox', 'scratch'),
         'mnv4n': ('mnv4', 'scratch')}
# arch -> allowed inits. 'persondet' = the vendor model's own weights (Ingenic: private); 'yolox' =
# Megvii's Apache-2.0 YOLOX-Nano COCO weights; 'mnv4' = timm's Apache-2.0 MobileNetV4 backbone +
# YOLOX-Nano's neck and heads; 'scratch' = random. Every init except 'persondet'
# goes through a float stage first (float_start_net -> pretrain_float -> make_qat_init).


def needs_float_stage(arch, init):
    return init != 'persondet'


def calib_images(root, n=32, split='train'):
    """n images (N x 3 x H x W, 0..255) spread over a split, for start-up calibration."""
    ds = DetSet(root, split, False)
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    return torch.stack([ds[i][0] for i in idx])


def float_start_net(arch, init, root=None):
    """The float network (BatchNorm on, quantisation off) the float stage starts from. root: the
    dataset, for archs whose start needs calibration images (mnv4n's offsets)."""
    if arch == 'mnv4n':
        import mnv4
        if init == 'mnv4':
            mp, yp = mnv4.weights_paths()
            return mnv4.from_mnv4(mp, yp, CLASSES, H, W, calib=calib_images(root) if root else None)
        return mnv4.scratch_float(CLASSES, H, W)
    if arch == 'yoloxn':
        import yoloxn
        if init == 'yolox':
            return yoloxn.from_yolox(yoloxn.weights_path(), CLASSES, H, W)
        return yoloxn.scratch_float(CLASSES, H, W)
    return scratch_detector(NC)


def build_net(arch, init='persondet'):
    """The network for arch, with its starting weights (init 'persondet' = the vendor model's own
    weights for that arch); any other init: the empty QAT form, to load a state (qat_init.pt, best.pt) into."""
    if arch == 'yoloxn':
        import yoloxn
        return yoloxn.build(NC, H, W)
    if arch == 'mnv4n':
        import mnv4
        return mnv4.build(NC, H, W)
    if arch == 'yolon':
        import jzdl
        from graphnet import from_jzdl
        src = jzdl.Model.parse(open(yolon_template_path(), 'rb').read())
        cmap = [YOLON_SRC_CLASSES.index(YOLON_FROM[c]) if c in YOLON_FROM else None for c in CLASSES]
        net = from_jzdl(src, NC, H, W, class_map=cmap)
        net.src_model = src
        return net
    return qat_net(NC) if init == 'scratch' else from_persondet(NC)


def export_any(net, path=None):
    """T31 .bin for either architecture."""
    if hasattr(net, 'spec'):
        from graphnet import export_graph
        return export_graph(net, net.src_model, path)
    return export(net, H, W, path)


def check_resume(ck, net, path):
    """Refuse to resume a checkpoint made for other classes (e.g. a v1 3-class run)."""
    if hasattr(net, 'spec'):
        k = 'convs.%d.weight' % net.head_layers[0]
        want, have = net.convs[str(net.head_layers[0])].weight.shape[0], ck['net'].get(k, torch.zeros(0)).shape[0]
    else:
        if 'L.l20.weight' not in ck['net']:
            raise SystemExit('%s belongs to a different architecture; use --name NAME' % path)
        want = net.L['l20'].weight.shape[0]
        have = ck['net']['L.l20.weight'].shape[0]
    if have != want:
        raise SystemExit('%s was made for %d classes, this build has %d (%s). Start a separate run '
                         'with --name NAME (images already downloaded are reused).'
                         % (path, have // 3 - 5, NC, ', '.join(CLASSES)))


def qat_net(nc):
    """An empty Detector in its QAT form (BatchNorm folded away), to load a QAT state into."""
    from net import Detector, QConv
    net = Detector(num_classes=nc)
    for m in net.modules():
        if isinstance(m, QConv):
            m.bn, m.bn_folded = None, True
    return net


def scratch_detector(nc, seed=0):
    """Randomly initialised Detector for float pre-training: BatchNorm on, quantisation off."""
    from net import Detector, QConv
    torch.manual_seed(seed)
    net = Detector(num_classes=nc)
    for m in net.modules():
        if isinstance(m, QConv):
            m.quant_w = False
            if m.act is not None:
                m.act.enabled = False
    with torch.no_grad():
        for h in ('l20', 'l26'):                        # objectness + class priors ~ 0.01, as YOLO does
            b = net.L[h].bias.view(3, 5 + nc)
            b[:, 4:] = -4.6
    return net


def _loaders(root, bs, limit, workers, dev, pkg_weight):
    use_cuda = _is_cuda(dev)
    tds = DetSet(root, 'train', True, limit, gpu_aug=use_cuda)
    if not len(tds):
        raise SystemExit('no training images under %s' % os.path.join(root, 'train', 'images'))
    wts = [pkg_weight if (PACKAGE is not None and len(lab) > 0 and np.any(lab[:, 0] == PACKAGE)) else 1.0
           for lab in tds.labels]
    sampler = torch.utils.data.WeightedRandomSampler(wts, len(wts), replacement=True)
    pin = use_cuda
    tr_kw = dict(prefetch_factor=4) if workers > 0 else {}
    tr = DataLoader(tds, bs, sampler=sampler, num_workers=workers, collate_fn=collate,
                    persistent_workers=workers > 0, drop_last=True, pin_memory=pin, **tr_kw)
    va_w = min(workers, 4)
    va_kw = dict(persistent_workers=va_w > 0, prefetch_factor=2) if va_w > 0 else {}
    va = DataLoader(DetSet(root, 'val', False, (limit // 5) if limit else 3000), bs,
                    num_workers=va_w, collate_fn=collate, pin_memory=pin, **va_kw)
    return tds, tr, va


def pretrain_float(root, out, epochs=120, bs=16, lr=1e-3, limit=None, workers=6, dev=None, pkg_weight=4.0,
                   progress=None, arch='persondet', init='scratch'):
    """Train arch in float (BatchNorm, no quantisation) from float_start_net(arch, init). Resumable
    (out/float_last.pt). Returns the trained float net."""
    os.makedirs(out, exist_ok=True)
    dev = pick_device(dev)
    use_cuda = _is_cuda(dev)
    use_bf16 = use_cuda and torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()
    net = float_start_net(arch, init, root)
    tds, tr, va = _loaders(root, bs, limit, workers, dev, pkg_weight)
    start, best, history = 0, -1.0, []
    ck_path = os.path.join(out, 'float_last.pt')
    ck = None
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location='cpu', weights_only=True)
        net.load_state_dict(ck['net'])
        start, history = ck['epoch'] + 1, ck['history']
        means = [h.get('mean') for h in history]
        best = max([float(v) for v in means if v is not None and v == v] + [-1.0])
    net = net.to(dev, memory_format=torch.channels_last if use_cuda else torch.preserve_format)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-4, fused=use_cuda)
    steps = epochs * len(tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: 0.5 * (1 + math.cos(math.pi * min(1, k / steps))) * min(1, (k + 1) / 1000))
    if ck is not None:
        opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        ck = None
    try:
        if use_cuda:
            # Float pre-training has no quantisation: unlock Blackwell/CUDA Tensor Cores & cuDNN autotuner.
            # setup_exact_math() is restored in finally: before BN folding & QAT.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        with open(os.path.join(out, 'log.txt'), 'a') as log:
            say = lambda m: print(m, file=log, flush=True)
            say('float stage (%s, init %s): %d imgs, device %s, bs %d, lr %g, epochs %d, start %d, bf16=%s'
                % (arch, init, len(tds), dev, bs, lr, epochs, start, use_bf16))
            if start == 0 and init != 'scratch':
                aps = evaluate(net, va, dev, amp_bf16=use_bf16, channels_last=use_cuda)
                say('float epoch -1 (%s init) AP50 %s' % (init, ' '.join('%s %.3f' % (c, v) for c, v in zip(CLASSES, aps))))
            info = dict(epochs=epochs, iters=len(tr), bs=bs, device=dev, history=history, best=best, phase='float')
            amp_ctx = (lambda: torch.amp.autocast('cuda', dtype=torch.bfloat16)) if use_bf16 else contextlib.nullcontext
            run_net = torch.compile(net) if (use_cuda and hasattr(net, 'spec') and steps >= 1000) else net
            for ep in range(start, epochs):
                net.train(); t0 = time.time()
                for k, (x, t) in enumerate(tr):
                    x = _gpu_color_aug(x.to(dev, non_blocking=True), channels_last=use_cuda)
                    t = t.to(dev, non_blocking=True)
                    with amp_ctx():
                        raw = run_net(x)
                    outs = {s: o.float() for s, o in head_dict(net, raw).items()}
                    loss, _ = yolo_loss(outs, t, anchors=anchors_of(net))
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                    opt.step(); sched.step()
                    if progress and (k % 20 == 0 or k == len(tr) - 1):
                        progress(dict(info, stage='float', epoch=ep, it=k + 1, img_s=(k + 1) * bs / (time.time() - t0),
                                      loss=loss.item(), t_epoch=time.time() - t0))
                    if k % 200 == 0:
                        say('float ep %d it %d/%d loss %.4f %.1f img/s' % (ep, k, len(tr), loss.item(), (k + 1) * bs / (time.time() - t0)))
                if ep % 10 == 9 or ep == epochs - 1:           # float AP now and then (cheap enough)
                    aps = evaluate(net, va, dev, amp_bf16=use_bf16, channels_last=use_cuda)
                    mean = float(np.nanmean(aps))
                    if mean == mean:
                        best = max(best, mean)
                    say('float epoch %d AP50 %s  mean %.3f' % (ep, ' '.join('%s %.3f' % (c, v) for c, v in zip(CLASSES, aps)), mean))
                    history.append(dict(epoch=ep, ap=dict(zip(CLASSES, [float(v) for v in aps])), mean=mean))
                sd = {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}
                torch.save({'net': sd, 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': ep,
                            'history': history}, ck_path + '.tmp')
                os.replace(ck_path + '.tmp', ck_path)
                info.update(history=history, best=best)
                if progress:
                    progress(dict(info, stage='epoch_done', epoch=ep, it=len(tr), t_epoch=time.time() - t0))
    finally:
        setup_exact_math()
    return net.to('cpu', memory_format=torch.contiguous_format), history


@torch.no_grad()
def make_qat_init(net, root, out, limit=None, n_batches=32, bs=16):
    """Float net -> QAT start: fold BatchNorm into the weights, calibrate every activation scale on
    training images, switch quantisation on. Writes out/qat_init.pt."""
    from calib import fold_bn, calibrate_acts, tame_collapsed_bn
    from net import QConv
    net = net.cpu().eval()
    tame_collapsed_bn(net)
    fold_bn(net)
    for m in net.modules():
        if isinstance(m, QConv):
            m.bn = None
    ds = DetSet(root, 'train', False, limit)
    idx = np.linspace(0, len(ds) - 1, n_batches * bs).astype(int)
    batches = [torch.stack([ds[i][0] for i in idx[k:k + bs]]) for k in range(0, len(idx), bs)]
    calibrate_acts(net, batches)
    from calib import fit_requant
    fit_requant(net)
    if hasattr(net, 'L'):
        repair_export_scales(net)
    qi_path = os.path.join(out, 'qat_init.pt')
    torch.save({'net': net.state_dict()}, qi_path + '.tmp')
    os.replace(qi_path + '.tmp', qi_path)
    return net


def train(root, out, epochs=30, bs=16, lr=5e-4, limit=None, workers=6, dev=None, pkg_weight=4.0,
          progress=None, init='persondet', arch='persondet'):
    """Quantisation-aware training. init='persondet' starts from Ingenic's persondet weights;
    init='scratch' starts from out/qat_init.pt (made by pretrain_float + make_qat_init).
    progress(dict) is called every few iterations and after each epoch (for status output)."""
    os.makedirs(out, exist_ok=True)
    dev = pick_device(dev)
    use_cuda = _is_cuda(dev)
    setup_exact_math()
    net = build_net(arch, init)
    if needs_float_stage(arch, init):
        net.load_state_dict(torch.load(os.path.join(out, 'qat_init.pt'), map_location='cpu', weights_only=True)['net'])
    tds, tr, va = _loaders(root, bs, limit, workers, dev, pkg_weight)   # package images oversampled
    start, best, history = 0, -1.0, []
    ck_path = os.path.join(out, 'last.pt')
    ck = None
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location='cpu', weights_only=True)
        check_resume(ck, net, ck_path)
        net.load_state_dict(ck['net'])
        start, best, history = ck['epoch'] + 1, float(ck.get('best', -1.0)), ck.get('history', [])
    net = net.to(dev)
    scales = [p for n, p in net.named_parameters() if n.endswith('log_s')]
    rest = [p for n, p in net.named_parameters() if not n.endswith('log_s')]
    opt = torch.optim.AdamW([{'params': rest, 'weight_decay': 1e-4},
                             {'params': scales, 'lr': lr * 0.1, 'weight_decay': 0}], lr=lr, fused=use_cuda)
    steps = epochs * len(tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: 0.5 * (1 + math.cos(math.pi * min(1, k / steps))) * min(1, (k + 1) / 300))
    if ck is not None and 'opt' in ck:
        opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
    ck = None
    with open(os.path.join(out, 'log.txt'), 'a') as log:
        say = lambda m: print(m, file=log, flush=True)
        say('train %d imgs, val %d, device %s, bs %d, lr %g, epochs %d, start %d' % (len(tds), len(va.dataset), dev, bs, lr, epochs, start))
        info = dict(epochs=epochs, iters=len(tr), bs=bs, device=dev, history=history, best=best)
        if start == 0:
            if hasattr(net, 'L'):
                repair_export_scales(net)
            else:
                from calib import fit_requant
                fit_requant(net)
            aps = evaluate(net, va, dev)
            say('epoch -1 (%s init) AP50 ' % (init if init != 'persondet' else 'Ingenic ' + arch) + '' + ' '.join('%s %.3f' % (c, v) for c, v in zip(CLASSES, aps)))
            history.append(dict(epoch=-1, ap=dict(zip(CLASSES, [float(v) for v in aps]))))
        for ep in range(start, epochs):
            net.train(); t0 = time.time()
            for k, (x, t) in enumerate(tr):
                x = _gpu_color_aug(x.to(dev, non_blocking=True))
                outs = head_dict(net, net(x))
                loss, _ = yolo_loss(outs, t.to(dev, non_blocking=True), anchors=anchors_of(net))
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step(); sched.step()
                if k % 20 == 0 or k == len(tr) - 1:
                    rate = (k + 1) * bs / (time.time() - t0)
                    if progress:
                        progress(dict(info, stage='train', epoch=ep, it=k + 1, img_s=rate, loss=loss.item(),
                                      t_epoch=time.time() - t0))
                if k % 200 == 0:
                    say('ep %d it %d/%d loss %.4f %.1f img/s' % (ep, k, len(tr), loss.item(), (k + 1) * bs / (time.time() - t0)))
            t_train = time.time() - t0
            if progress:
                progress(dict(info, stage='eval', epoch=ep, it=len(tr), t_epoch=t_train))
            if hasattr(net, 'L'):
                repair_export_scales(net)
            else:
                from calib import fit_requant
                fit_requant(net)
            aps = evaluate(net, va, dev)
            mean = float(np.nanmean(aps))
            say('epoch %d AP50 %s  mean %.3f  (%.0f s)' % (ep, ' '.join('%s %.3f' % (c, v) for c, v in zip(CLASSES, aps)),
                                                          mean, time.time() - t0))
            history.append(dict(epoch=ep, ap=dict(zip(CLASSES, [float(v) for v in aps])), mean=mean, seconds=round(time.time() - t0),
                                ap_sized={k: {ck_: float(cv_) for ck_, cv_ in v.items()} for k, v in LAST_EVAL.items()}))
            say('         AP50 objects >= 32 px: %s' % ' '.join('%s %.3f' % kv for kv in LAST_EVAL['medium_large'].items()))
            export_any(net.cpu(), os.path.join(out, 'model_ep%02d.bin' % ep))
            sd_cpu = {k: v.detach().clone() for k, v in net.state_dict().items()}
            if mean > best:
                best = mean
                bp = os.path.join(out, 'best.pt')
                torch.save({'net': sd_cpu, 'epoch': ep, 'mean': mean, 'init': init, 'arch': arch,
                            'classes': CLASSES}, bp + '.tmp')
                os.replace(bp + '.tmp', bp)
                export_any(net, os.path.join(out, 'model_best.bin'))
            net.to(dev)
            torch.save({'net': sd_cpu, 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': ep,
                        'best': best, 'history': history}, ck_path + '.tmp')
            os.replace(ck_path + '.tmp', ck_path)
            info.update(history=history, best=best)
            if progress:
                progress(dict(info, stage='epoch_done', epoch=ep, it=len(tr), t_epoch=time.time() - t0))
    return history, best


def main(a):
    arg = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d
    train(a[1], arg('--out', os.path.join(HERE, '..', 'runs', time.strftime('%Y%m%d-%H%M'))),
          epochs=arg('--epochs', 30), bs=arg('--bs', 16), lr=arg('--lr', 5e-4), limit=arg('--limit', 0) or None,
          workers=arg('--workers', 6), dev=arg('--device', '') or None, pkg_weight=arg('--pkg-weight', 4.0))


if __name__ == '__main__':
    main(sys.argv)
