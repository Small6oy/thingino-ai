#!/usr/bin/env python3
"""QAT training of the person / pet / package detector (hardware-exact forward).

  train.py DATA_ROOT [--epochs 30] [--bs 16] [--lr 5e-4] [--out runs/x] [--limit N]
                     [--workers N] [--device cuda|mps|cpu] [--pkg-weight 4.0]

Normally run through ../build_model.py. Re-running with the same --out resumes from
last.pt (model, optimiser and schedule).

DATA_ROOT/{train,val}/{images,labels} from data_prep.py. Starts from Ingenic's persondet
(init_persondet.py), heads 3 x (5 + 3). Geometry matches raptor/aitest: a random 16:9
crop stretched to 416x320, RGB. Decode/anchors match aitest headed_decode (YOLOv5 style,
yolov3-tiny anchors). Each epoch: eval mAP@0.5 per class and export a .bin.
"""
import glob, math, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from init_persondet import from_persondet  # noqa: E402
from export import export  # noqa: E402

H, W = 320, 416
NC = 3
CLASSES = ['person', 'pet', 'package']
ANCHORS = {32: [(81, 82), (135, 169), (344, 319)], 16: [(10, 14), (23, 27), (37, 58)]}
BALANCE = {16: 1.0, 32: 0.4}


# ------------------------------------------------------------------ data
def label_path(img):
    """split/images/x.jpg -> split/labels/x.txt (works with either path separator)."""
    d, f = os.path.split(img)
    return os.path.join(os.path.dirname(d), 'labels', os.path.splitext(f)[0] + '.txt')


class DetSet(Dataset):
    def __init__(self, root, split, train, limit=None):
        self.imgs = sorted(glob.glob(os.path.join(root, split, 'images', '*.jpg')))
        if limit:
            random.Random(0).shuffle(self.imgs); self.imgs = self.imgs[:limit]
        self.train = train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        from PIL import Image, ImageEnhance
        p = self.imgs[i]
        im = Image.open(p).convert('RGB')
        lp = label_path(p)
        lab = np.loadtxt(lp, ndmin=2).reshape(-1, 5) if os.path.getsize(lp) else np.zeros((0, 5))
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


def yolo_loss(outs, targets, nc=NC):
    """outs: {stride: (N, 3*(5+nc), gh, gw)} ; targets (n, 6): img, cls, cx, cy, w, h (pixels)"""
    dev = targets.device
    lbox = torch.zeros(1, device=dev); lobj = torch.zeros(1, device=dev); lcls = torch.zeros(1, device=dev)
    bce = torch.nn.BCEWithLogitsLoss()
    off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=dev).float() * 0.5
    for s, p in outs.items():
        n, _, gh, gw = p.shape
        p = p.view(n, 3, 5 + nc, gh, gw).permute(0, 1, 3, 4, 2)       # n, a, y, x, k
        tobj = torch.zeros(p.shape[:4], device=dev)
        anc = torch.tensor(ANCHORS[s], device=dev).float() / s       # in grid units
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
            gxy = ta[:, 2:4]; gxi = torch.tensor([gw, gh], device=dev) - gxy
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
def decode(outs, conf=0.05, nc=NC):
    """-> per image list of (x0, y0, x1, y1, score, cls), aitest headed_decode semantics"""
    res = None
    for s, p in outs.items():
        n, _, gh, gw = p.shape
        p = p.view(n, 3, 5 + nc, gh, gw).permute(0, 1, 3, 4, 2).sigmoid()
        anc = torch.tensor(ANCHORS[s], device=p.device).float()
        gy, gx = torch.meshgrid(torch.arange(gh, device=p.device), torch.arange(gw, device=p.device), indexing='ij')
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
        h32, h16 = net(x.to(dev))
        dets += [d.cpu() for d in decode({32: h32.float(), 16: h16.float()})]
        for i in range(len(x)):
            gts.append(t[t[:, 0] == i][:, 1:])
    return ap50(dets, gts)


# ------------------------------------------------------------------ main
def pick_device(want=None):
    if want:
        return want
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def setup_exact_math():
    """The fake-quant forward relies on exact float32 integer arithmetic: no TF32."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def train(root, out, epochs=30, bs=16, lr=5e-4, limit=None, workers=6, dev=None, pkg_weight=4.0,
          progress=None):
    """progress(dict) is called every few iterations and after each epoch (for status output)."""
    os.makedirs(out, exist_ok=True)
    dev = pick_device(dev)
    setup_exact_math()
    net = from_persondet(NC)
    tds = DetSet(root, 'train', True, limit)
    if not len(tds):
        raise SystemExit('no training images under %s' % os.path.join(root, 'train', 'images'))
    # oversample package images (Open Images "Box" is only ~2.2k images vs ~20k person)
    wts = []
    for p in tds.imgs:
        lp = label_path(p)
        has_pkg = os.path.getsize(lp) > 0 and any(l.startswith('2 ') for l in open(lp))
        wts.append(pkg_weight if has_pkg else 1.0)
    sampler = torch.utils.data.WeightedRandomSampler(wts, len(wts), replacement=True)
    pin = dev == 'cuda'
    tr = DataLoader(tds, bs, sampler=sampler, num_workers=workers, collate_fn=collate,
                    persistent_workers=workers > 0, drop_last=True, pin_memory=pin)
    va = DataLoader(DetSet(root, 'val', False, (limit // 5) if limit else 3000), bs,
                    num_workers=min(workers, 4), collate_fn=collate, pin_memory=pin)
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
        net.load_state_dict(ck['net'])
        if 'opt' in ck:
            opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        start, best, history = ck['epoch'] + 1, ck.get('best', -1.0), ck.get('history', [])
    net = net.to(dev)
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(dev)
    log = open(os.path.join(out, 'log.txt'), 'a')
    say = lambda m: print(m, file=log, flush=True)
    say('train %d imgs, val %d, device %s, bs %d, lr %g, epochs %d, start %d' % (len(tds), len(va.dataset), dev, bs, lr, epochs, start))
    info = dict(epochs=epochs, iters=len(tr), bs=bs, device=dev, history=history, best=best)
    if start == 0:
        aps = evaluate(net, va, dev)
        say('epoch -1 (persondet init) AP50 ' + ' '.join('%s %.3f' % (c, v) for c, v in zip(CLASSES, aps)))
        history.append(dict(epoch=-1, ap=dict(zip(CLASSES, aps))))
    for ep in range(start, epochs):
        net.train(); t0 = time.time()
        for k, (x, t) in enumerate(tr):
            h32, h16 = net(x.to(dev, non_blocking=True))
            loss, _ = yolo_loss({32: h32, 16: h16}, t.to(dev))
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
        export(net.cpu(), H, W, os.path.join(out, 'model_ep%02d.bin' % ep))
        if mean > best:
            best = mean
            torch.save({'net': net.state_dict(), 'epoch': ep, 'mean': mean}, os.path.join(out, 'best.pt'))
            export(net, H, W, os.path.join(out, 'model_best.bin'))
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
