#!/usr/bin/env python3
"""Build the person / pet / package detector for Thingino T31 cameras (JZDL .bin), end to end.

    python build_model.py            # full build: check, download, self-test, train, verify, package
    python build_model.py --quick    # 10-minute smoke test on a small subset (do this first)

Every step is resumable: run the same command again after an interruption and it picks up
where it stopped (downloaded images are kept, training resumes from the last epoch).

Options:
    --work DIR        working directory (default: train/work)  - needs ~4 GB free
    --epochs N        default 30
    --bs N --lr X     batch size / learning rate (default: 64 / 1e-3 on CUDA, 16 / 5e-4 otherwise)
    --workers N       data-loading processes (default: CPU cores - 2, max 12)
    --device D        cuda | mps | cpu (default: best available)
    --jobs N          parallel image downloads (default 32)
    --name NAME       run name (default: full, or quick with --quick); same name = resume
    --limit N         use only N training images (testing)
    --force           continue even if the GPU self-test is not exact

Status: a live line in the terminal, plus WORK/runs/NAME/status.json (stage, epoch, speed,
ETA, accuracy so far) that anyone can read while it runs. Results land in train/out/NAME/.
"""
import gzip, hashlib, http.client, io, json, os, platform, shutil, sys, threading, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'qat'))
sys.path.insert(0, os.path.join(HERE, '..', 'tools', 'jzdl'))

SOURCES = json.load(open(os.path.join(HERE, 'sources.json')))
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
        import numpy, PIL, torch, torchvision  # noqa: F401
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
def get_template():
    t = SOURCES['template']
    path = os.path.join(HERE, 'cache', 'magik_model_persondet.bin')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not (os.path.exists(path) and hashlib.md5(open(path, 'rb').read()).hexdigest() == t['md5']):
        print('  downloading persondet (%d KB) from gtxaspec/magik_models' % (t['bytes'] // 1024))
        data = fetch(t['url'])
        if hashlib.md5(data).hexdigest() != t['md5']:
            raise SystemExit('persondet download has the wrong md5 (%s); upstream file changed?' % hashlib.md5(data).hexdigest())
        open(path + '.tmp', 'wb').write(data)
        os.replace(path + '.tmp', path)
    print('  persondet OK (md5 %s)' % t['md5'])
    os.environ['JZDL_TEMPLATE'] = path
    return path


# ------------------------------------------------------------------ 3. dataset
def manifest(split):
    rows = []
    with gzip.open(os.path.join(HERE, 'manifest', split + '.tsv.gz'), 'rt') as f:
        for line in f:
            line = line.rstrip('\n')
            if line:
                name, _, labels = line.partition('\t')
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


def get_dataset(root, jobs, limit, status):
    counts = {}
    for split in ('train', 'val'):
        rows = manifest(split)
        if limit:                                    # spread over the list so every source/class is sampled
            n = limit if split == 'train' else max(200, limit // 5)
            rows = [rows[i * len(rows) // n] for i in range(min(n, len(rows)))]
        for d in ('images', 'labels'):
            os.makedirs(os.path.join(root, split, d), exist_ok=True)
        todo = [r for r in rows if not os.path.exists(os.path.join(root, split, 'images', r[0] + '.jpg'))]
        ok = len(rows) - len(todo); bad = 0; t0 = time.time()
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
            h32, h16 = net(torch.from_numpy(rgb).permute(2, 0, 1)[None].float())
        ref = jzdl_ref.forward(model, rgb)
        for th, blob in ((h32, 22), (h16, 28)):
            p = np.ascontiguousarray(th[0].permute(1, 2, 0).numpy())
            r = ref[blob]
            exact &= np.array_equal(r.view(np.uint32), p.view(np.uint32))
            worst = max(worst, float(np.abs(r - p).max()))
    return exact, worst


def selftest(root, dev, force):
    import numpy as np, torch, jzdl
    from init_persondet import from_persondet
    from export import export
    from train import setup_exact_math, H as TH, W as TW
    setup_exact_math()
    net = from_persondet(3)
    m, rep = export(net, TH, TW)
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
        ap = ('  AP50 person %.2f pet %.2f package %.2f' % tuple(last['ap'][c] for c in ('person', 'pet', 'package'))
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
        status.update(force=p['stage'] == 'epoch_done', stage='train', epoch=ep + 1, epochs=epochs, it=it, iters=iters,
                      img_s=round(p.get('img_s') or 0, 1), eta_s=round(eta) if eta is not None else None,
                      elapsed_s=round(time.time() - t_start), best_mean_ap50=round(p['best'], 4), history=p['history'])
    return progress


# ------------------------------------------------------------------ main
def main(a):
    quick = '--quick' in a
    work = os.path.abspath(arg(a, '--work', os.path.join(HERE, 'work')))
    os.makedirs(work, exist_ok=True)
    name = arg(a, '--name', None) or ('quick' if quick else 'full')
    run_dir = os.path.join(work, 'runs', name)
    os.makedirs(run_dir, exist_ok=True)
    status = Status(os.path.join(run_dir, 'status.json'))
    t0 = time.time()
    total = 6
    print('thingino-ai model build: person / pet / package for T31 (JZDL)%s' % ('  [QUICK smoke test]' if quick else ''))
    print('run %s, work dir %s' % (name, work))

    step(1, total, 'checking the machine')
    dev, gpu = check_env(a, work)
    cuda = dev == 'cuda'
    epochs = arg(a, '--epochs', 1 if quick else 30)
    bs = arg(a, '--bs', 64 if cuda else 16)
    lr = arg(a, '--lr', 1e-3 if cuda else 5e-4)
    workers = arg(a, '--workers', max(2, min(12, (os.cpu_count() or 4) - 2)))
    limit = arg(a, '--limit', 2000 if quick else 0) or None
    status.update(force=True, stage='check', device=dev, gpu=gpu, epochs=epochs, bs=bs, lr=lr)

    step(2, total, 'persondet starting weights')
    get_template()

    step(3, total, 'dataset (COCO person/cat/dog + Open Images Box, ~2.5 GB)')
    data_root = os.path.join(work, 'dataset')
    counts = get_dataset(data_root, arg(a, '--jobs', 32), limit, status)

    step(4, total, 'self-test: exact integer arithmetic')
    status.update(force=True, stage='selftest')
    st = selftest(data_root, dev, '--force' in a)

    step(5, total, 'training: %d epochs, batch %d, lr %g on %s (%s)' % (epochs, bs, lr, dev, gpu or dev))
    print('  live status: %s' % status.path)
    from train import train
    history, best = train(data_root, run_dir, epochs=epochs, bs=bs, lr=lr, limit=limit, workers=workers, dev=dev,
                          progress=make_progress(status, epochs))

    step(6, total, 'verifying and packaging the best model')
    import torch, jzdl
    from init_persondet import from_persondet
    status.update(force=True, stage='verify')
    ck = torch.load(os.path.join(run_dir, 'best.pt'), map_location='cpu')
    net = from_persondet(3); net.load_state_dict(ck['net'])
    model_path = os.path.join(run_dir, 'model_best.bin')
    data = open(model_path, 'rb').read()
    exact, worst = bitwise_check(net, jzdl.Model.parse(data), val_frames(data_root, 4))
    print('  model_best.bin (epoch %d) == trained PyTorch model, bit for bit: %s' % (ck['epoch'] + 1, exact))
    out = os.path.join(HERE, 'out', name)
    os.makedirs(out, exist_ok=True)
    shutil.copy(model_path, os.path.join(out, 'model_best.bin'))
    shutil.copy(os.path.join(run_dir, 'log.txt'), os.path.join(out, 'log.txt'))
    import numpy, torchvision
    report = dict(
        name=name, quick=quick, finished=time.strftime('%Y-%m-%d %H:%M:%S'), total_time=fmt_t(time.time() - t0),
        machine=dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                     torchvision=torchvision.__version__, numpy=numpy.__version__, device=dev, gpu=gpu,
                     cpus=os.cpu_count()),
        settings=dict(epochs=epochs, bs=bs, lr=lr, workers=workers, limit=limit),
        dataset=counts, selftest=st,
        best=dict(epoch=ck['epoch'], mean_ap50=round(best, 4), model_md5=hashlib.md5(data).hexdigest(),
                  bytes=len(data), bit_exact=exact, max_diff=worst),
        history=history)
    json.dump(clean(report), open(os.path.join(out, 'report.json'), 'w'), indent=1)
    status.update(force=True, stage='done', out=out)
    b = history[[h['epoch'] for h in history].index(ck['epoch'])]['ap']
    print('\nDone in %s. Best epoch %d: AP50 person %.3f, pet %.3f, package %.3f.'
          % (fmt_t(time.time() - t0), ck['epoch'] + 1, b['person'], b['pet'], b['package']))
    print('Please send us the three files in:\n  %s\n(model_best.bin, report.json, log.txt; about 1 MB).'
          ' Don\'t publish model_best.bin: it is derived from Ingenic\'s weights.' % out)


if __name__ == '__main__':
    try:
        main(sys.argv)
    except KeyboardInterrupt:
        print('\nstopped. Run the same command again to resume.')
        sys.exit(130)
