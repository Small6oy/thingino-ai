#!/usr/bin/env python3
"""Build the person / pet / package detector for Thingino cameras, end to end: one trained model,
exported twice (T31: JZDL .bin, bit-exact; T41: float ONNX -> Ingenic converter -> NNA .bin).

    python build_model.py            # full build: check, download, self-test, train, verify, package
    python build_model.py --quick    # 10-minute smoke test on a small subset (do this first)

Every step is resumable: run the same command again after an interruption and it picks up
where it stopped (downloaded images are kept, training resumes from the last epoch).

Options:
    --work DIR        working directory (default: train/work)  - needs ~4 GB free
    --epochs N        default 30
    --bs N --lr X     batch size / learning rate (default: 64 / 1e-3 on CUDA and XPU, 16 / 5e-4 otherwise)
    --workers N       data-loading processes (default: CPU cores - 2, max 12)
    --device D        cuda | xpu | mps | cpu (default: best available)
    --jobs N          parallel image downloads (default 32)
    --name NAME       run name (default: full, or quick with --quick); same name = resume
    --limit N         use only N training images (testing)
    --force           continue even if the GPU self-test is not exact
    --calib DIR       T41 calibration frames (your camera's day + IR snapshots; default: validation images)
    --t41-convert     also try the T41 converter on a non-x86-64 machine (Docker with amd64 emulation)
    --t41-eval N      validation frames for the int8 NNA simulation check (default 600, 0 = skip)
    --init I          persondet (default: fine-tune Ingenic's persondet; models stay private) or
                      scratch (random start, float pre-training then QAT; no Ingenic weights, so the
                      models can be shared). Default run name: full / full-scratch.
    --min-box PX      leave boxes smaller than PX (sqrt(w*h) at 416x320) out of the loss (default 12, 0 = off)
    --float-epochs N  float pre-training epochs with --init scratch (default 120)
    --float-lr X      float pre-training learning rate (default 2e-3 on CUDA/XPU, 1e-3 otherwise)

Status: a live line in the terminal, plus WORK/runs/NAME/status.json (stage, epoch, speed,
ETA, accuracy so far) that anyone can read while it runs. Results land in train/out/NAME/:
model_t31.bin, model_t41.onnx (+ model_t41.bin when Docker on x86-64 is available), report.json, log.txt,
checkpoint.pt.
"""
import gzip, hashlib, http.client, io, json, os, platform, shutil, sys, threading, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'qat'))
sys.path.insert(0, os.path.join(HERE, '..', 'tools', 'jzdl'))

SOURCES = json.load(open(os.path.join(HERE, 'sources.json')))
CLASSES = json.load(open(os.path.join(HERE, 'manifest', 'classes.json')))['classes']   # same list as qat/train.py


def aps_str(ap):
    return '  '.join('%s %.3f' % (c, ap.get(c, float('nan'))) for c in CLASSES)
UA = {'User-Agent': 'thingino-ai-build/1.0', 'Connection': 'keep-alive'}
_TLS = threading.local()
H, W = 320, 416
MAX_SIDE = 640
TTY = sys.stdout.isatty()


# ------------------------------------------------------------------ output helpers
def arg(a, k, d):
    if k not in a:
        return d
    v = a[a.index(k) + 1]
    return v if d is None else type(d)(v)


def step(n, total, title):
    print('\n[%d/%d] %s' % (n, total, title), flush=True)


def fmt_t(s):
    s = int(max(0, s))
    return '%dh%02dm' % (s // 3600, s % 3600 // 60) if s >= 3600 else '%dm%02ds' % (s // 60, s % 60)


class Status:
    """Live terminal line + status.json."""

    def __init__(self, path):
        self.path, self.data, self.last_line, self.last_write = path, {}, 0.0, 0.0

    KEEP = ('device', 'gpu', 'epochs', 'bs', 'lr')

    def update(self, force=False, **kw):
        if kw.get('stage', self.data.get('stage')) != self.data.get('stage'):
            self.data = {k: v for k, v in self.data.items() if k in self.KEEP}      # drop the old stage's fields
        self.data.update(kw, updated=time.strftime('%Y-%m-%d %H:%M:%S'))
        now = time.time()
        if force or now - self.last_write > 5:
            tmp = self.path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(clean(self.data), f, indent=1)
            os.replace(tmp, self.path)
            self.last_write = now

    def line(self, text, force=False):
        now = time.time()
        if TTY:
            if force or now - self.last_line > 1:
                print('\r' + text[:shutil.get_terminal_size((120, 20)).columns - 1].ljust(
                    shutil.get_terminal_size((120, 20)).columns - 1), end='', flush=True)
                self.last_line = now
        elif force or now - self.last_line > 60:
            print(text, flush=True)
            self.last_line = now

    def done(self):
        if TTY:
            print(flush=True)


def clean(o):
    """NaN (a class with no validation boxes) -> null, so the JSON stays valid."""
    if isinstance(o, float):
        return None if o != o else round(o, 4)
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def _get_conn(scheme, netloc):
    pool = getattr(_TLS, 'conns', None)
    if pool is None:
        pool = _TLS.conns = {}
    key = (scheme, netloc)
    conn = pool.get(key)
    if conn is None:
        cls = http.client.HTTPSConnection if scheme == 'https' else http.client.HTTPConnection
        conn = cls(netloc, timeout=60)
        pool[key] = conn
    return conn


def _drop_conn(scheme, netloc):
    pool = getattr(_TLS, 'conns', None)
    if pool is not None:
        conn = pool.pop((scheme, netloc), None)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass


def _fetch_once(url, redirects=5):
    cur = url
    for _ in range(redirects + 1):
        u = urllib.parse.urlsplit(cur)
        path = u.path or '/'
        if u.query:
            path += '?' + u.query
        conn = _get_conn(u.scheme, u.netloc)
        try:
            conn.request('GET', path, headers=UA)
            resp = conn.getresponse()
            data = resp.read()
        except Exception:
            _drop_conn(u.scheme, u.netloc)
            raise
        if resp.status in (301, 302, 303, 307, 308):
            loc = resp.getheader('Location')
            if not loc:
                raise http.client.HTTPException('HTTP %d without Location' % resp.status)
            cur = urllib.parse.urljoin(cur, loc)
            continue
        if resp.status != 200:
            raise http.client.HTTPException('HTTP %d for %s' % (resp.status, cur))
        if resp.will_close:
            _drop_conn(u.scheme, u.netloc)
        return data
    raise http.client.HTTPException('too many redirects for %s' % url)


def fetch(url, tries=4):
    """Fetch URL bytes using per-thread HTTP(S) Keep-Alive connections."""
    err = None
    for k in range(tries):
        try:
            return _fetch_once(url)
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(1.5 * (k + 1))
    raise err


# ------------------------------------------------------------------ 1. environment
def cuda_arch_ok(cap, arch_list):
    """sm_XY code runs on the same major with minor >= Y (sm_120 on a DGX Spark's sm_121);
    compute_XY (PTX) is JIT-compiled for anything >= XY."""
    have = cap[0] * 10 + cap[1]
    for a in arch_list:
        kind, _, v = a.partition('_')
        if not v.isdigit():
            continue
        v = int(v)
        if kind == 'sm' and v // 10 == cap[0] and v <= have:
            return True
        if kind == 'compute' and v <= have:
            return True
    return False


def check_env(a, work):
    try:
        import numpy, PIL, torch, torchvision, onnx  # noqa: F401
    except ImportError as e:
        raise SystemExit('missing Python package: %s\nInstall the requirements first, see README.md.' % e.name)
    import torch
    from train import pick_device
    dev = arg(a, '--device', None) or pick_device()
    print('  python %s, torch %s, torchvision %s, %s' % (platform.python_version(), torch.__version__,
                                                         __import__('torchvision').__version__, platform.platform()))
    gpu = None
    if dev == 'cuda':
        if not torch.cuda.is_available():
            raise SystemExit('--device cuda but torch.cuda.is_available() is False (CPU-only torch build?). See README.md.')
        cap = torch.cuda.get_device_capability(0)
        gpu = torch.cuda.get_device_name(0)
        arch = 'sm_%d%d' % cap
        print('  GPU: %s (%s, %.1f GB)' % (gpu, arch, torch.cuda.get_device_properties(0).total_memory / 2 ** 30))
        if not cuda_arch_ok(cap, torch.cuda.get_arch_list()):
            raise SystemExit('this torch build has no kernels for %s (%s). RTX 50xx / DGX Spark need a CUDA 12.8+ build: '
                             'see README.md.' % (arch, ', '.join(torch.cuda.get_arch_list())))
        x = torch.ones(1, 3, 8, 8, device='cuda')            # fails early on driver / kernel problems
        torch.nn.functional.conv2d(x, torch.ones(4, 3, 3, 3, device='cuda')).sum().item()
    elif dev.startswith('xpu'):
        if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
            raise SystemExit('--device %s but torch.xpu.is_available() is False (no Intel XPU torch build?). See README.md.' % dev)
        idx = int(dev.partition(':')[2] or 0)
        gpu = torch.xpu.get_device_name(idx)
        print('  GPU: %s (Intel XPU, %.1f GB)' % (gpu, torch.xpu.get_device_properties(idx).total_memory / 2 ** 30))
        x = torch.ones(1, 3, 8, 8, device=dev)               # fails early on driver / kernel problems
        torch.nn.functional.conv2d(x, torch.ones(4, 3, 3, 3, device=dev)).sum().item()
    elif dev == 'mps':
        print('  GPU: Apple MPS')
    else:
        print('  no GPU found: training on the CPU will take a very long time (days).')
    free = shutil.disk_usage(work).free / 2 ** 30
    print('  free disk in %s: %.1f GB' % (work, free))
    if free < 4:
        raise SystemExit('need about 4 GB free for the images and checkpoints')
    return dev, gpu


# ------------------------------------------------------------------ 2. persondet template
def get_template(key='template', fname='magik_model_persondet.bin', env='JZDL_TEMPLATE', what='persondet'):
    t = SOURCES[key]
    path = os.path.join(HERE, 'cache', fname)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not (os.path.exists(path) and hashlib.md5(open(path, 'rb').read()).hexdigest() == t['md5']):
        print('  downloading %s (%d KB) from gtxaspec/magik_models' % (what, t['bytes'] // 1024))
        data = fetch(t['url'])
        if hashlib.md5(data).hexdigest() != t['md5']:
            raise SystemExit('%s download has the wrong md5 (%s); upstream file changed?' % (what, hashlib.md5(data).hexdigest()))
        open(path + '.tmp', 'wb').write(data)
        os.replace(path + '.tmp', path)
    print('  %s OK (md5 %s)' % (what, t['md5']))
    os.environ[env] = path
    return path


# ------------------------------------------------------------------ 3. dataset
def manifest(split):
    rows = []
    with gzip.open(os.path.join(HERE, 'manifest', split + '.tsv.gz'), 'rt') as f:
        for line in f:
            line = line.rstrip('\n')
            if line:
                name, _, labels = line.partition('\t')
                if not name or '/' in name or '\\' in name or '..' in name:
                    raise SystemExit('invalid manifest entry name: %r' % (name,))
                rows.append((name, labels.split('|') if labels else []))
    return rows


def image_url(split, name):
    src, _, ident = name.partition('_')
    u = SOURCES['images']
    if src == 'coco':
        return u['coco'].format(split=split, id=ident)
    return u['oi_' + split].format(id=ident)


def save_one(split, root, name, labels):
    from PIL import Image
    img = os.path.join(root, split, 'images', name + '.jpg')
    lbl = os.path.join(root, split, 'labels', name + '.txt')
    if os.path.exists(img):
        return True
    try:
        im = Image.open(io.BytesIO(fetch(image_url(split, name)))).convert('RGB')
    except Exception:  # noqa: BLE001
        return False
    s = MAX_SIDE / max(im.size)
    if s < 1:
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BILINEAR)
    with open(lbl, 'w') as f:                       # label first: an image never exists without one
        f.write(''.join(l + '\n' for l in labels))
    im.save(img + '.tmp', 'JPEG', quality=90)
    os.replace(img + '.tmp', img)
    return True


def get_bundle(key):
    """Download a pinned image bundle (sources.json "bundles") once into cache/, md5-checked."""
    import hashlib, urllib.request
    b = SOURCES['bundles'][key]
    os.makedirs(os.path.join(HERE, 'cache'), exist_ok=True)
    path = os.path.join(HERE, 'cache', os.path.basename(urllib.parse.urlparse(b['url']).path))
    if os.path.exists(path) and os.path.getsize(path) == b['bytes']:
        return path
    print('  downloading the %s image bundle (%d MB, once)' % (key, b['bytes'] >> 20))
    h = hashlib.md5()
    with urllib.request.urlopen(b['url'], timeout=120) as r, open(path + '.part', 'wb') as f:
        for chunk in iter(lambda: r.read(1 << 20), b''):
            f.write(chunk); h.update(chunk)
    if h.hexdigest() != b['md5']:
        raise SystemExit('%s bundle md5 mismatch: delete %s and run again' % (key, path + '.part'))
    os.replace(path + '.part', path)
    return path


def unpack_bundle(key, root, split, rows):
    """Put the bundle's images for rows (name, labels) into root/split, with their manifest labels."""
    import tarfile
    want = {r[0]: r[1] for r in rows}
    n = 0
    with tarfile.open(get_bundle(key)) as t:
        for m in t:
            name = os.path.basename(m.name)[:-4]
            if not m.isfile() or not m.name.endswith('.jpg') or name not in want:
                continue
            with open(os.path.join(root, split, 'labels', name + '.txt'), 'w') as f:
                f.write(''.join(l + '\n' for l in want[name]))
            img = os.path.join(root, split, 'images', name + '.jpg')
            with t.extractfile(m) as src, open(img + '.tmp', 'wb') as dst:
                shutil.copyfileobj(src, dst)
            os.replace(img + '.tmp', img); n += 1
    return n


def refresh_labels(root, split, rows):
    """Labels always come from the manifest: images already on disk (e.g. from a v1 build in the same
    work dir) get their label files rewritten when the manifest changed them. Also refuses a manifest
    with a class id the model doesn't have (that would crash training with 'index out of bounds')."""
    nc, fixed = len(CLASSES), 0
    for name, labels in rows:
        for l in labels:
            c = int(l.split()[0])
            if not 0 <= c < nc:
                raise SystemExit('manifest row %s has class %d, but classes.json lists %d classes (%s). '
                                 'Run git pull in the thingino-ai folder so code and manifest match.'
                                 % (name, c, nc, ', '.join(CLASSES)))
        lbl = os.path.join(root, split, 'labels', name + '.txt')
        if not os.path.exists(os.path.join(root, split, 'images', name + '.jpg')):
            continue
        want = ''.join(l + '\n' for l in labels)
        try:
            have = open(lbl).read()
        except OSError:
            have = None
        if have != want:
            open(lbl, 'w').write(want); fixed += 1
    if fixed:
        print('  %s: refreshed %d label files from the manifest (images kept)' % (split, fixed))


def get_dataset(root, jobs, limit, status):
    counts = {}
    for split in ('train', 'val'):
        rows = manifest(split)
        if limit:                                    # spread over the list so every source/class is sampled
            n = limit if split == 'train' else max(200, limit // 5)
            rows = [rows[i * len(rows) // n] for i in range(min(n, len(rows)))]
        for d in ('images', 'labels'):
            os.makedirs(os.path.join(root, split, d), exist_ok=True)
        refresh_labels(root, split, rows)
        todo = [r for r in rows if not os.path.exists(os.path.join(root, split, 'images', r[0] + '.jpg'))]
        rf = [r for r in todo if r[0].startswith('rf_')]
        if rf:                                       # Roboflow Universe images come from our pinned bundle
            unpack_bundle('roboflow', root, split, rf)
        miss_rf = sum(not os.path.exists(os.path.join(root, split, 'images', r[0] + '.jpg')) for r in rf)
        todo = [r for r in todo if not r[0].startswith('rf_')]
        ok = len(rows) - len(todo) - miss_rf; bad = miss_rf; t0 = time.time()
        if todo:
            print('  %s: %d images, %d already here, downloading %d (COCO + Open Images servers)'
                  % (split, len(rows), ok, len(todo)))
        with ThreadPoolExecutor(jobs) as ex:
            for k, good in enumerate(ex.map(lambda r: save_one(split, root, *r), todo)):
                ok += good; bad += not good
                rate = (k + 1) / max(1e-6, time.time() - t0)
                txt = '  %s %d/%d  %.0f img/s  failed %d  ETA %s' % (split, ok + bad, len(rows), rate, bad,
                                                                      fmt_t((len(todo) - k - 1) / rate))
                status.line(txt, force=k == len(todo) - 1)
                status.update(stage='download', split=split, done=ok + bad, total=len(rows), failed=bad,
                              eta_s=round((len(todo) - k - 1) / rate))
        status.done()
        print('  %s: %d of %d images ready' % (split, ok, len(rows)))
        if bad > 0.03 * len(rows):
            raise SystemExit('%d downloads failed (> 3%%). Check the connection and run again; finished images are kept.' % bad)
        with open(os.path.join(root, split, 'index.txt'), 'w') as f:   # exactly this build's images
            f.writelines(r[0] + '\n' for r in rows if os.path.exists(os.path.join(root, split, 'images', r[0] + '.jpg')))
        counts[split] = dict(images=ok, missing=len(rows) - ok)
    return counts


# ------------------------------------------------------------------ 4/6. exactness checks
def val_frames(root, n):
    import numpy as np
    from train import DetSet
    ds = DetSet(root, 'val', False)
    step_ = max(1, len(ds) // n)
    return [np.ascontiguousarray(ds[i * step_][0].permute(1, 2, 0).numpy().astype(np.uint8)) for i in range(n)]


def bitwise_check(net, model, frames):
    """PyTorch (CPU) fake-quant net == integer reference of the exported .bin, bit for bit."""
    import numpy as np, torch, jzdl_ref
    net = net.eval().cpu()
    worst, exact = 0.0, True
    for rgb in frames:
        with torch.no_grad():
            outs = net(torch.from_numpy(rgb).permute(2, 0, 1)[None].float())
        ref = jzdl_ref.forward(model, rgb)
        blobs = [model.layers[i].tops[0] for i in net.head_layers] if hasattr(net, 'head_layers') else [22, 28]
        for th, blob in zip(outs, blobs):
            p = np.ascontiguousarray(th[0].permute(1, 2, 0).numpy())
            r = ref[blob]
            exact &= np.array_equal(r.view(np.uint32), p.view(np.uint32))
            worst = max(worst, float(np.abs(r - p).max()))
    return exact, worst


def selftest(root, dev, force, init='persondet', arch='persondet'):
    import numpy as np, torch, jzdl
    from export import export
    from train import setup_exact_math, H as TH, W as TW, build_net, export_any
    setup_exact_math()
    if arch != 'persondet':
        net = build_net(arch)
    elif init == 'scratch':                         # random weights, BatchNorm folded, scales calibrated
        from train import scratch_detector
        from calib import fold_bn, calibrate_acts
        net = scratch_detector(len(CLASSES)).eval()
        fold_bn(net)
        for c in net.L.values():
            c.bn = None
        calibrate_acts(net, [torch.from_numpy(np.stack(val_frames(root, 4))).permute(0, 3, 1, 2).float()])
    else:
        from init_persondet import from_persondet
        net = from_persondet(len(CLASSES))
    m, rep = export_any(net)
    assert jzdl.Model.parse(m.to_bytes()).to_bytes() == m.to_bytes()
    frames = val_frames(root, 3)
    exact, worst = bitwise_check(net, m, frames)
    print('  export -> .bin -> integer reference == PyTorch: %s (max diff %g)' % ('bit-exact' if exact else 'NOT exact', worst))
    if not exact:
        raise SystemExit('the integer pipeline is not exact on this machine; please send us this output')
    dev_diff = 0.0
    if dev != 'cpu':
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
        with torch.no_grad():
            c = net.cpu()(x)
            g = net.to(dev)(x.to(dev))
        dev_diff = max(float((a - b.cpu()).abs().max()) for a, b in zip(c, g))
        ok = dev_diff < 1e-3
        print('  %s forward vs CPU: max logit diff %.2g (%s)' % (dev, dev_diff, 'OK' if ok else 'TOO LARGE'))
        if not ok and not force:
            raise SystemExit('the GPU forward does not match the CPU one; re-run with --device cpu, or --force to train anyway')
    return dict(bit_exact=exact, device_vs_cpu_max_diff=dev_diff, acc_bound_max=int(max(rep.values())))


# ------------------------------------------------------------------ 5. training with status
def make_progress(status, epochs):
    t_start = time.time()
    epoch_secs = []

    def progress(p):
        ep, it, iters = p['epoch'], p['it'], p['iters']
        if p['stage'] == 'epoch_done':
            epoch_secs.append(p['t_epoch'])
        per_epoch = (sum(epoch_secs) / len(epoch_secs)) if epoch_secs else (
            iters * p['bs'] / p['img_s'] * 1.08 if p.get('img_s') else None)
        if per_epoch:
            left_this = 0 if p['stage'] == 'epoch_done' else max(0.0, per_epoch - p['t_epoch'])
            eta = left_this + (epochs - ep - 1) * per_epoch
        else:
            eta = None
        hist = [h for h in p['history'] if h['epoch'] >= 0]
        last = hist[-1] if hist else None
        ap = ('  AP50 ' + ' '.join('%s %.2f' % (c, v) for c, v in last['ap'].items())
              if last else '')
        if p['stage'] == 'epoch_done':
            status.line('  epoch %d/%d done in %s.%s  best mean %.3f  ETA %s'
                        % (ep + 1, epochs, fmt_t(p['t_epoch']), ap, p['best'], fmt_t(eta) if eta is not None else '?'), force=True)
            status.done()
        else:
            status.line('  epoch %d/%d  it %d/%d  %s  %s  ETA %s%s' % (
                ep + 1, epochs, it, iters,
                'evaluating' if p['stage'] == 'eval' else '%.0f img/s  loss %.3f' % (p['img_s'], p['loss']),
                fmt_t(time.time() - t_start), fmt_t(eta) if eta is not None else '?', ap))
        status.update(force=p['stage'] == 'epoch_done', stage=p.get('phase', 'train'), epoch=ep + 1, epochs=epochs, it=it, iters=iters,
                      img_s=round(p.get('img_s') or 0, 1), eta_s=round(eta) if eta is not None else None,
                      elapsed_s=round(time.time() - t_start), best_mean_ap50=round(p['best'], 4), history=p['history'])
    return progress


# ------------------------------------------------------------------ 7. T41 export + conversion
MAGIK_IMAGE = 'thingino-ai-magik:2.0.4'


def calib_frames(a, data_root, n=100):
    """Calibration set for the NNA's int8 quantiser: --calib DIR (your camera's frames, day + IR,
    recommended) or n validation images. Each is cropped to 16:9 and stretched to 416x320 like raptor."""
    import glob, numpy as np
    from PIL import Image
    from train import H as TH, W as TW
    d = arg(a, '--calib', None)
    if not d:
        return val_frames(data_root, n), 'validation images'
    files = sorted(f for e in ('jpg', 'jpeg', 'png') for f in glob.glob(os.path.join(d, '*.' + e)))
    if not files:
        raise SystemExit('--calib %s: no .jpg/.png files' % d)
    out = []
    for f in files[:400]:
        im = Image.open(f).convert('RGB'); iw, ih = im.size
        cw, ch = (ih * 16 / 9, ih) if iw / ih > 16 / 9 else (iw, iw * 9 / 16)
        x0, y0 = (iw - cw) / 2, (ih - ch) / 2
        im = im.crop((round(x0), round(y0), round(x0 + cw), round(y0 + ch))).resize((TW, TH), Image.BILINEAR)
        out.append(np.asarray(im, np.uint8))
    return out, d


def convert_t41(t41_dir, force=False):
    """Run Ingenic's converter in Docker (x86-64 only). Returns (path or None, message)."""
    import subprocess
    if not shutil.which('docker'):
        return None, 'skipped: no docker on this machine (we convert it on our side)'
    if platform.machine().lower() not in ('x86_64', 'amd64') and not force:
        return None, 'skipped: the converter is an x86-64 program (%s here; we convert it on our side)' % platform.machine()
    if subprocess.run(['docker', 'image', 'inspect', MAGIK_IMAGE], capture_output=True).returncode != 0:
        print('  building the converter image (downloads Ingenic\'s tool, ~1 GB with Ubuntu, once)')
        r = subprocess.run(['docker', 'build', '-t', MAGIK_IMAGE, os.path.join(HERE, 't41')])
        if r.returncode:
            return None, 'converter image build failed'
    user = ['-u', '%d:%d' % (os.getuid(), os.getgid())] if hasattr(os, 'getuid') else []
    cmd = ('magik-transform-tools --inputpath model_t41.onnx --outputpath model_t41.mk.h --config magik_t41.cfg '
           '--save_quantize_model true > convert.log 2>&1; '
           'cp save-magik/model_quant.mgk model_t41_quant.mgk 2>/dev/null; rm -rf ./*.mk.h save-magik; true')
    print('  converting (post-training int8 quantisation, ~2 min)')
    subprocess.run(['docker', 'run', '--rm', '-e', 'HOME=/tmp', '-v', '%s:/work' % os.path.abspath(t41_dir)] + user
                   + [MAGIK_IMAGE, 'sh', '-c', cmd])
    b = os.path.join(t41_dir, 'model_t41.bin')          # the tool exits 1 even on success: judge by the file
    if os.path.exists(b) and os.path.getsize(b) > 0:
        return b, 'ok'
    tail = open(os.path.join(t41_dir, 'convert.log'), errors='replace').read()[-600:] if os.path.exists(
        os.path.join(t41_dir, 'convert.log')) else ''
    return None, 'converter produced no .bin; log tail: ' + tail.replace('\n', ' | ')


def eval_t41_int8(t41_dir, net, data_root, n, limit):
    """Score the converted model with Ingenic's PC executor (int8 simulation of the NNA), next to the
    float ONNX twin and the exact T31 model, on the same n validation frames."""
    import subprocess, numpy as np, torch
    from concurrent.futures import ThreadPoolExecutor
    from export_t41 import FloatDetector, FloatGraph, head_names
    from train import DetSet, decode, ap50, CLASSES, head_dict, anchors_of
    ds = DetSet(data_root, 'val', False, (limit // 5) if limit else 3000)
    n = min(n, len(ds))
    frames, gts = [], []
    for i in np.linspace(0, len(ds) - 1, n).astype(int):
        x, t = ds[i]
        frames.append(x.permute(1, 2, 0).numpy().astype(np.uint8)); gts.append(t[:, 1:])
    F_ = np.stack(frames)
    jobs = max(1, min(16, (os.cpu_count() or 2) - 1, n))
    ed = os.path.join(t41_dir, 'eval')
    os.makedirs(ed, exist_ok=True)
    shards = np.array_split(np.arange(n), jobs)
    for j, sh in enumerate(shards):
        F_[sh].tofile(os.path.join(ed, 's%d.u8' % j))
    user = ['-u', '%d:%d' % (os.getuid(), os.getgid())] if hasattr(os, 'getuid') else []
    t0 = time.time()
    print('  int8 simulation of the NNA on %d validation frames (%d parallel, ~2.5 s per frame each)' % (n, jobs))

    def run(j):
        return subprocess.run(['docker', 'run', '--rm', '-v', '%s:/work' % os.path.abspath(t41_dir)] + user
                              + [MAGIK_IMAGE, 'mgkrun', 'model_t41_quant.mgk', str(F_.shape[1]), str(F_.shape[2]),
                                 'eval/s%d.u8' % j, 'eval/s%d' % j], capture_output=True, text=True)
    with ThreadPoolExecutor(jobs) as ex:
        res = list(ex.map(run, range(jobs)))
    if any(r.returncode for r in res):
        shutil.rmtree(ed, ignore_errors=True)
        return dict(error='mgkrun failed: ' + res[0].stderr[-300:])
    shapes, heads = {}, head_names(net)
    for l in res[0].stdout.splitlines():
        p = l.split()
        if len(p) == 5:
            for h in heads:
                if p[0] == h or p[0].startswith(h + '_'):
                    shapes[h] = (p[0], [int(v) for v in p[2:]])

    def load(k):
        name, (hh, ww, cc) = shapes[k]
        return torch.from_numpy(np.concatenate([np.fromfile(os.path.join(ed, 's%d.%s.f32' % (j, name)), np.float32)
                                                .reshape(-1, hh, ww, cc) for j in range(jobs)])).permute(0, 3, 1, 2)
    qs = [load(h) for h in heads]
    shutil.rmtree(ed, ignore_errors=True)
    x = torch.from_numpy(F_).permute(0, 3, 1, 2).float()
    net = net.eval().cpu(); fd = FloatGraph(net) if hasattr(net, 'spec') else FloatDetector(net)
    anchors = anchors_of(net)

    def run_net(m):
        with torch.no_grad():
            o = [m(x[k:k + 32]) for k in range(0, n, 32)]
        return [torch.cat([p[j] for p in o]) for j in range(len(heads))]

    def score(outs):
        dets = []
        for k in range(0, n, 64):
            dets += decode(head_dict(net, [o[k:k + 64] for o in outs]), anchors=anchors)
        return dict(zip(CLASSES, ap50(dets, gts)))
    fs = run_net(fd)
    cos = {h: float(torch.nn.functional.cosine_similarity(a.flatten(1), b.flatten(1)).mean()) for h, a, b in zip(heads, qs, fs)}
    rep = dict(frames=n, seconds=round(time.time() - t0), cosine_int8_vs_float=cos,
               ap50=dict(t31_exact=score(run_net(net)), t41_float=score(fs), t41_int8=score(qs)))
    print('  int8 vs float outputs: cosine ' + ', '.join('%s %.4f' % kv for kv in cos.items()) + ' (Ingenic: > 0.98 is good)')
    for k, lab in (('t31_exact', 'T31 exact'), ('t41_float', 'T41 float'), ('t41_int8', 'T41 int8 (NNA sim)')):
        v = rep['ap50'][k]
        print('  %-19s AP50 %s   (%d frames)' % (lab, aps_str(v), n))
    return rep


def export_t41_stage(a, net, data_root, out, dev, limit):
    import numpy as np
    from torch.utils.data import DataLoader
    from export_t41 import export_t41, FloatDetector, FloatGraph
    from train import DetSet, collate, evaluate, CLASSES, H as TH, W as TW
    t41_dir = os.path.join(out, 't41')
    frames, src = calib_frames(a, data_root)
    rep = export_t41(net, TH, TW, t41_dir, calib_frames=frames, check_frames=val_frames(data_root, 4))
    shutil.copy(os.path.join(t41_dir, 'model_t41.onnx'), os.path.join(out, 'model_t41.onnx'))
    d = rep.get('onnx_vs_torch_max_diff')
    print('  model_t41.onnx: %.1f MB, opset %d, ONNX == PyTorch float twin: %s' % (
        rep['onnx_bytes'] / 2 ** 20, rep['opset'], 'max diff %.1g' % d if d is not None else 'not checked (no onnxruntime)'))
    print('  calibration set: %d frames from %s' % (len(frames), src))
    va = DataLoader(DetSet(data_root, 'val', False, (limit // 5) if limit else 3000), 32, num_workers=2, collate_fn=collate)
    aps = evaluate((FloatGraph(net) if hasattr(net, 'spec') else FloatDetector(net)).to(dev), va, dev)
    rep['float_ap50'] = dict(zip(CLASSES, aps))
    print('  float AP50 (what the NNA approximates): ' + '  '.join('%s %.3f' % (c, v) for c, v in zip(CLASSES, aps)))
    b, msg = convert_t41(t41_dir, '--t41-convert' in a)
    print('  T41 .bin: ' + (msg if not b else '%s (%d KB)' % (b, os.path.getsize(b) // 1024)))
    rep['convert'] = msg
    if b:
        shutil.copy(b, os.path.join(out, 'model_t41.bin'))
        rep['bin'] = dict(bytes=os.path.getsize(b), md5=hashlib.md5(open(b, 'rb').read()).hexdigest())
        if os.path.exists(os.path.join(t41_dir, 'model_t41_quant.mgk')) and arg(a, '--t41-eval', 600) > 0:
            rep['int8_eval'] = eval_t41_int8(t41_dir, net, data_root, arg(a, '--t41-eval', 600), limit)
    rep['calib_source'] = 'custom' if arg(a, '--calib', None) else 'validation images'
    return rep


# ------------------------------------------------------------------ main
def main(a):
    quick = '--quick' in a
    work = os.path.abspath(arg(a, '--work', os.path.join(HERE, 'work')))
    os.makedirs(work, exist_ok=True)
    init = arg(a, '--init', 'persondet')
    arch = arg(a, '--arch', 'persondet')
    if init not in ('persondet', 'scratch'):
        raise SystemExit('--init must be persondet or scratch')
    if arch not in ('persondet', 'yolon'):
        raise SystemExit('--arch must be persondet or yolon')
    if arch == 'yolon' and init == 'scratch':
        raise SystemExit('--arch yolon starts from Ingenic\'s YOLOv5 sample; --init scratch is persondet-only for now')
    name = arg(a, '--name', None) or (('quick' if quick else 'full') + ('-scratch' if init == 'scratch' else '')
                                      + ('-yolon' if arch == 'yolon' else ''))
    run_dir = os.path.join(work, 'runs', name)
    os.makedirs(run_dir, exist_ok=True)
    status = Status(os.path.join(run_dir, 'status.json'))
    t0 = time.time()
    total = 7
    print('thingino-ai model build: ' + ' / '.join(CLASSES) + ', one model for T31 (JZDL) + T41 (NNA)%s' % ('  [QUICK smoke test]' if quick else ''))
    print('run %s, work dir %s, start: %s' % (name, work, 'from scratch (float pre-training, then QAT)'
                                                if init == 'scratch' else "Ingenic's persondet weights"))

    step(1, total, 'checking the machine')
    dev, gpu = check_env(a, work)
    accel = dev == 'cuda' or dev.startswith('xpu')
    epochs = arg(a, '--epochs', 1 if quick else 30)
    bs = arg(a, '--bs', 64 if accel else 16)
    lr = arg(a, '--lr', 1e-3 if accel else 5e-4)
    workers = arg(a, '--workers', max(2, min(16, (os.cpu_count() or 4) - 2)))
    limit = arg(a, '--limit', 2000 if quick else 0) or None
    status.update(force=True, stage='check', device=dev, gpu=gpu, epochs=epochs, bs=bs, lr=lr)

    if init == 'scratch':
        step(2, total, 'starting weights: none (training from scratch)')
        print('  not needed: the model is trained from random weights on open data only')
    elif arch == 'yolon':
        step(2, total, 'YOLO-N starting weights (Ingenic YOLOv5 JZDL sample)')
        get_template('template_yolon', 'magik_model_yolov5.bin', 'JZDL_YOLON_TEMPLATE', 'YOLOv5 sample')
    else:
        step(2, total, 'persondet starting weights')
        get_template()

    step(3, total, 'dataset (COCO + Open Images, %d classes, ~4.5 GB)' % len(CLASSES))
    data_root = os.path.join(work, 'dataset')
    counts = get_dataset(data_root, arg(a, '--jobs', 32), limit, status)

    step(4, total, 'self-test: exact integer arithmetic')
    status.update(force=True, stage='selftest')
    st = selftest(data_root, dev, '--force' in a, init, arch)

    step(5, total, 'training: %d epochs, batch %d, lr %g on %s (%s)' % (epochs, bs, lr, dev, gpu or dev))
    print('  live status: %s' % status.path)
    import train as train_mod
    train_mod.MIN_BOX_TRAIN[0] = arg(a, '--min-box', train_mod.MIN_BOX)
    from train import train
    if init == 'scratch':
        from train import pretrain_float, make_qat_init
        fe = arg(a, '--float-epochs', 1 if quick else 120)
        if not os.path.exists(os.path.join(run_dir, 'qat_init.pt')):
            print('  phase 1: float pre-training, %d epochs (then BatchNorm folding + activation calibration)' % fe)
            fnet, _ = pretrain_float(data_root, run_dir, epochs=fe, bs=bs, lr=arg(a, '--float-lr', 2e-3 if accel else 1e-3),
                                     limit=limit, workers=workers, dev=dev, progress=make_progress(status, fe))
            status.done()
            make_qat_init(fnet, data_root, run_dir, limit)
        print('  phase 2: quantisation-aware training, %d epochs' % epochs)
    history, best = train(data_root, run_dir, epochs=epochs, bs=bs, lr=lr, limit=limit, workers=workers, dev=dev,
                          progress=make_progress(status, epochs), init=init, arch=arch)

    step(6, total, 'T31 export: verify the JZDL model bit for bit')
    import torch, jzdl
    from train import build_net
    status.update(force=True, stage='export_t31')
    ck = torch.load(os.path.join(run_dir, 'best.pt'), map_location='cpu', weights_only=True)
    net = build_net(arch, 'scratch' if arch == 'persondet' else 'persondet')
    net.load_state_dict(ck['net']); net.eval()
    with open(os.path.join(run_dir, 'model_best.bin'), 'rb') as f:
        data = f.read()
    exact, worst = bitwise_check(net, jzdl.Model.parse(data), val_frames(data_root, 4))
    print('  model_t31.bin (epoch %d) == trained PyTorch model, bit for bit: %s' % (ck['epoch'] + 1, exact))
    out = os.path.join(HERE, 'out', name)
    os.makedirs(out, exist_ok=True)
    t31_out = os.path.join(out, 'model_t31.bin')
    with open(t31_out + '.tmp', 'wb') as f:
        f.write(data)
    os.replace(t31_out + '.tmp', t31_out)
    shutil.copy(os.path.join(run_dir, 'log.txt'), os.path.join(out, 'log.txt'))
    shutil.copy(os.path.join(run_dir, 'best.pt'), os.path.join(out, 'checkpoint.pt'))   # re-export source (T41, later fixes)

    step(7, total, 'T41 export: float ONNX for the NNA converter')
    status.update(force=True, stage='export_t41')
    t41 = export_t41_stage(a, net, data_root, out, dev, limit)

    import numpy, torchvision
    report = dict(
        name=name, quick=quick, finished=time.strftime('%Y-%m-%d %H:%M:%S'), total_time=fmt_t(time.time() - t0),
        machine=dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                     torchvision=torchvision.__version__, numpy=numpy.__version__, device=dev, gpu=gpu,
                     cpus=os.cpu_count()),
        settings=dict(epochs=epochs, bs=bs, lr=lr, workers=workers, limit=limit, init=init, arch=arch,
                      min_box=arg(a, '--min-box', 12),
                      float_epochs=arg(a, '--float-epochs', 1 if quick else 120) if init == 'scratch' else 0),
        classes=CLASSES,
        dataset=counts, selftest=st,
        best=dict(epoch=ck['epoch'], mean_ap50=round(best, 4), model_md5=hashlib.md5(data).hexdigest(),
                  bytes=len(data), bit_exact=exact, max_diff=worst),
        t41=t41, history=history)
    rep_path = os.path.join(out, 'report.json')
    with open(rep_path + '.tmp', 'w') as f:
        json.dump(clean(report), f, indent=1)
    os.replace(rep_path + '.tmp', rep_path)
    status.update(force=True, stage='done', out=out)
    b = history[[h['epoch'] for h in history].index(ck['epoch'])]['ap']
    print('\nDone in %s. Best epoch %d, AP50:' % (fmt_t(time.time() - t0), ck['epoch'] + 1))
    print('  T31 (exact)        %s' % aps_str(b))
    bs_ = history[[h['epoch'] for h in history].index(ck['epoch'])].get('ap_sized', {}).get('medium_large')
    if bs_:
        print('    objects >= 32 px %s   (what a camera alert usually cares about)' % aps_str(bs_))
    print('  T41 float ONNX     %s   (before the NNA\'s int8 conversion)' % aps_str(t41['float_ap50']))
    q = (t41.get('int8_eval') or {}).get('ap50', {}).get('t41_int8')
    if q:
        print('  T41 int8 (NNA sim) %s   (%d-frame subset, see report.json)' % (aps_str(q), t41['int8_eval']['frames']))
    send = (['model_t31.bin', 'model_t41.onnx'] + (['model_t41.bin'] if t41.get('bin') else [])
            + ['report.json', 'log.txt', 'checkpoint.pt'])
    print('Please send us these files from\n  %s\n  %s' % (out, ', '.join(send)))
    if init == 'scratch':
        print('This model was trained from scratch on open data only (no Ingenic weights).')
    else:
        print('Don\'t publish the models or the checkpoint: they are derived from Ingenic\'s %s weights.' % ('YOLOv5 sample' if arch == 'yolon' else 'persondet'))

if __name__ == '__main__':
    try:
        main(sys.argv)
    except KeyboardInterrupt:
        print('\nstopped. Run the same command again to resume.')
        sys.exit(130)
