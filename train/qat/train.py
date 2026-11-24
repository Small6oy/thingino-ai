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
import glob, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from init_persondet import from_persondet  # noqa: E402
from export import export  # noqa: E402

H, W = 320, 416
# class list: train/manifest/classes.json (written by data/build_manifest.py); v1 had 3 classes
_CJ = os.path.join(HERE, '..', 'manifest', 'classes.json')
CLASSES = json.load(open(_CJ))['classes'] if os.path.exists(_CJ) else ['person', 'pet', 'package']
NC = len(CLASSES)
ANCHORS = {32: [(81, 82), (135, 169), (344, 319)], 16: [(10, 14), (23, 27), (37, 58)]}
BALANCE = {8: 4.0, 16: 1.0, 32: 0.4}                 # objectness weight per stride (YOLOv5's P3/P4/P5)


def head_dict(net, outs):
    """Net outputs -> {stride: logits}. Detector (persondet) returns (h32, h16); a GraphNet
    returns its heads in output order with net.strides."""
    strides = getattr(net, 'strides', None) or [32, 16]
    return dict(zip(strides, outs))


def anchors_of(net):
    return getattr(net, 'anchors', None) or ANCHORS


# ------------------------------------------------------------------ data
def label_path(img):
    """split/images/x.jpg -> split/labels/x.txt (works with either path separator)."""
    d, f = os.path.split(img)
    return os.path.join(os.path.dirname(d), 'labels', os.path.splitext(f)[0] + '.txt')


def _load_labels(lp):
    """Fast plain-Python YOLO label parser (matches np.loadtxt(..., ndmin=2).reshape(-1, 5))."""
    rows = []
    with open(lp, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 5:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5) if rows else np.zeros((0, 5), dtype=np.float64)


class DetSet(Dataset):
    def __init__(self, root, split, train, limit=None):
        self.imgs = sorted(glob.glob(os.path.join(root, split, 'images', '*.jpg')))
        if limit:
            random.Random(0).shuffle(self.imgs); self.imgs = self.imgs[:limit]
        self.labels = [_load_labels(label_path(p)) for p in self.imgs]
        self.train = train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        from PIL import Image, ImageEnhance
        p = self.imgs[i]
        im = Image.open(p).convert('RGB')
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
            im = ImageEnhance.Brightness(im).enhance(random.uniform(0.6, 1.4))
            im = ImageEnhance.Contrast(im).enhance(random.uniform(0.7, 1.3))
            im = ImageEnhance.Color(im).enhance(random.uniform(0.5, 1.5))
            if random.random() < 0.25:                       # IR night mode: grayscale
                im = im.convert('L').convert('RGB')
        x = torch.from_numpy(np.asarray(im, np.uint8).copy()).permute(2, 0, 1).float()
        t = torch.zeros((len(b), 6))
        if len(b):
            t[:, 1] = torch.from_numpy(cls).float()
            t[:, 2:] = torch.from_numpy(np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                                                  b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)).float()
        return x, t


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
            b = ta[:, 0].long()
            ps = p[b, ai, gj, gi]
            pxy = ps[:, :2].sigmoid() * 2 - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anc[ai]
            tbox = torch.cat((ta[:, 2:4] - torch.stack([gi, gj], 1).float(), ta[:, 4:6]), 1)
            iou = bbox_ciou(torch.cat((pxy, pwh), 1), tbox)
            lbox += (1 - iou).mean()
            tobj[b, ai, gj, gi] = iou.detach().clamp(0).to(tobj.dtype)
            tc = torch.zeros((len(ps), nc), device=dev); tc[range(len(ps)), ta[:, 1].long()] = 1
            lcls += bce(ps[:, 5:], tc)
        lobj += bce(p[..., 4], tobj) * BALANCE[s]
    return lbox * 0.05 + lobj * 1.0 + lcls * 0.5, torch.cat((lbox, lobj, lcls)).detach()


# ------------------------------------------------------------------ decode + eval (as aitest)
def decode(outs, conf=0.05, nc=NC, anchors=None):
    """-> per image list of (x0, y0, x1, y1, score, cls), aitest headed_decode semantics"""
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
    out = []
    for d in res:
        d = d[d[:, 4] > conf]
        keep = []
        for c in range(nc):
            dc = d[d[:, 5] == c]
            if len(dc):
                from torchvision.ops import nms
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


@torch.no_grad()
def evaluate(net, loader, dev):
    net.eval()
    dets, gts = [], []
    for x, t in loader:
        outs = {s: o.float() for s, o in head_dict(net, net(x.to(dev))).items()}
        dets += [d.cpu() for d in decode(outs, anchors=anchors_of(net))]
        for i in range(len(x)):
            gts.append(t[t[:, 0] == i][:, 1:])
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


def build_net(arch, init='persondet'):
    """The network for arch, with its starting weights (init 'persondet' = the vendor model's own
    weights for that arch; 'scratch' only for persondet: empty QAT form, loaded from a state)."""
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
    tds = DetSet(root, 'train', True, limit)
    if not len(tds):
        raise SystemExit('no training images under %s' % os.path.join(root, 'train', 'images'))
    wts = [pkg_weight if (len(lab) > 0 and np.any(lab[:, 0] == 2)) else 1.0 for lab in tds.labels]
    sampler = torch.utils.data.WeightedRandomSampler(wts, len(wts), replacement=True)
    pin = dev == 'cuda'
    tr = DataLoader(tds, bs, sampler=sampler, num_workers=workers, collate_fn=collate,
                    persistent_workers=workers > 0, drop_last=True, pin_memory=pin)
    va = DataLoader(DetSet(root, 'val', False, (limit // 5) if limit else 3000), bs,
                    num_workers=min(workers, 4), collate_fn=collate, pin_memory=pin)
    return tds, tr, va


def pretrain_float(root, out, epochs=120, bs=16, lr=1e-3, limit=None, workers=6, dev=None, pkg_weight=4.0,
                   progress=None):
    """Train the Detector from random weights in float (BatchNorm, no quantisation). Resumable
    (out/float_last.pt). Returns the trained float net."""
    os.makedirs(out, exist_ok=True)
    dev = pick_device(dev)
    net = scratch_detector(NC)
    tds, tr, va = _loaders(root, bs, limit, workers, dev, pkg_weight)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-4)
    steps = epochs * len(tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: 0.5 * (1 + math.cos(math.pi * min(1, k / steps))) * min(1, (k + 1) / 1000))
    start, history = 0, []
    ck_path = os.path.join(out, 'float_last.pt')
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location='cpu')
        net.load_state_dict(ck['net']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        start, history = ck['epoch'] + 1, ck['history']
    net = net.to(dev)
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(dev)
    with open(os.path.join(out, 'log.txt'), 'a') as log:
        say = lambda m: print(m, file=log, flush=True)
        say('float pre-training from scratch: %d imgs, device %s, bs %d, lr %g, epochs %d, start %d'
            % (len(tds), dev, bs, lr, epochs, start))
        info = dict(epochs=epochs, iters=len(tr), bs=bs, device=dev, history=history, best=-1.0, phase='float')
        for ep in range(start, epochs):
            net.train(); t0 = time.time()
            for k, (x, t) in enumerate(tr):
                outs = head_dict(net, net(x.to(dev, non_blocking=True)))
                loss, _ = yolo_loss(outs, t.to(dev), anchors=anchors_of(net))
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step(); sched.step()
                if progress and (k % 20 == 0 or k == len(tr) - 1):
                    progress(dict(info, stage='float', epoch=ep, it=k + 1, img_s=(k + 1) * bs / (time.time() - t0),
                                  loss=loss.item(), t_epoch=time.time() - t0))
                if k % 200 == 0:
                    say('float ep %d it %d/%d loss %.4f %.1f img/s' % (ep, k, len(tr), loss.item(), (k + 1) * bs / (time.time() - t0)))
            if ep % 10 == 9 or ep == epochs - 1:           # float AP now and then (cheap enough)
                aps = evaluate(net, va, dev)
                say('float epoch %d AP50 %s  mean %.3f' % (ep, ' '.join('%s %.3f' % (c, v) for c, v in zip(CLASSES, aps)),
                                                           float(np.nanmean(aps))))
                history.append(dict(epoch=ep, ap=dict(zip(CLASSES, aps)), mean=float(np.nanmean(aps))))
            torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': ep,
                        'history': history}, ck_path + '.tmp')
            os.replace(ck_path + '.tmp', ck_path)
            info.update(history=history)
            if progress:
                progress(dict(info, stage='epoch_done', epoch=ep, it=len(tr), t_epoch=time.time() - t0))
    return net.cpu(), history


@torch.no_grad()
def make_qat_init(net, root, out, limit=None, n_batches=32, bs=16):
    """Float net -> QAT start: fold BatchNorm into the weights, calibrate every activation scale on
    training images, switch quantisation on. Writes out/qat_init.pt."""
    from calib import fold_bn, calibrate_acts
    from net import QConv
    net = net.cpu().eval()
    fold_bn(net)
    for m in net.modules():
        if isinstance(m, QConv):
            m.bn = None
    ds = DetSet(root, 'train', False, limit)
    idx = np.linspace(0, len(ds) - 1, n_batches * bs).astype(int)
    batches = [torch.stack([ds[i][0] for i in idx[k:k + bs]]) for k in range(0, len(idx), bs)]
    calibrate_acts(net, batches)
    torch.save({'net': net.state_dict()}, os.path.join(out, 'qat_init.pt'))
    return net


def train(root, out, epochs=30, bs=16, lr=5e-4, limit=None, workers=6, dev=None, pkg_weight=4.0,
          progress=None, init='persondet', arch='persondet'):
    """Quantisation-aware training. init='persondet' starts from Ingenic's persondet weights;
    init='scratch' starts from out/qat_init.pt (made by pretrain_float + make_qat_init).
    progress(dict) is called every few iterations and after each epoch (for status output)."""
    os.makedirs(out, exist_ok=True)
    dev = pick_device(dev)
    setup_exact_math()
    net = build_net(arch, init)
    if init == 'scratch':
        net.load_state_dict(torch.load(os.path.join(out, 'qat_init.pt'), map_location='cpu')['net'])
    tds, tr, va = _loaders(root, bs, limit, workers, dev, pkg_weight)   # package images oversampled
    scales = [p for n, p in net.named_parameters() if n.endswith('log_s')]
    rest = [p for n, p in net.named_parameters() if not n.endswith('log_s')]
    opt = torch.optim.AdamW([{'params': rest, 'weight_decay': 1e-4},
                             {'params': scales, 'lr': lr * 0.1, 'weight_decay': 0}], lr=lr)
    steps = epochs * len(tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: 0.5 * (1 + math.cos(math.pi * min(1, k / steps))) * min(1, (k + 1) / 300))
    start, best, history = 0, -1.0, []
    ck_path = os.path.join(out, 'last.pt')
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location='cpu')
        check_resume(ck, net, ck_path)
        net.load_state_dict(ck['net'])
        if 'opt' in ck:
            opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        start, best, history = ck['epoch'] + 1, ck.get('best', -1.0), ck.get('history', [])
    net = net.to(dev)
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(dev)
    with open(os.path.join(out, 'log.txt'), 'a') as log:
        say = lambda m: print(m, file=log, flush=True)
        say('train %d imgs, val %d, device %s, bs %d, lr %g, epochs %d, start %d' % (len(tds), len(va.dataset), dev, bs, lr, epochs, start))
        info = dict(epochs=epochs, iters=len(tr), bs=bs, device=dev, history=history, best=best)
        if start == 0:
            aps = evaluate(net, va, dev)
            say('epoch -1 (%s init) AP50 ' % init + '' + ' '.join('%s %.3f' % (c, v) for c, v in zip(CLASSES, aps)))
            history.append(dict(epoch=-1, ap=dict(zip(CLASSES, aps))))
        for ep in range(start, epochs):
            net.train(); t0 = time.time()
            for k, (x, t) in enumerate(tr):
                outs = head_dict(net, net(x.to(dev, non_blocking=True)))
                loss, _ = yolo_loss(outs, t.to(dev), anchors=anchors_of(net))
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
            aps = evaluate(net, va, dev)
            mean = float(np.nanmean(aps))
            say('epoch %d AP50 %s  mean %.3f  (%.0f s)' % (ep, ' '.join('%s %.3f' % (c, v) for c, v in zip(CLASSES, aps)),
                                                          mean, time.time() - t0))
            history.append(dict(epoch=ep, ap=dict(zip(CLASSES, aps)), mean=mean, seconds=round(time.time() - t0)))
            export_any(net.cpu(), os.path.join(out, 'model_ep%02d.bin' % ep))
            if mean > best:
                best = mean
                torch.save({'net': net.state_dict(), 'epoch': ep, 'mean': mean, 'init': init, 'arch': arch,
                            'classes': CLASSES}, os.path.join(out, 'best.pt'))
                export_any(net, os.path.join(out, 'model_best.bin'))
            net.to(dev)
            torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': ep,
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
