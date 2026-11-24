#!/usr/bin/env python3
"""Fetch open Roboflow Universe datasets (parcels at doors, CCTV pets, ...) into our class scheme.

  python3 data/fetch_roboflow.py OUT [--key-file ~/.config/roboflow/key] [--only NAME,...]

For each dataset in DATASETS it asks the Roboflow API for the project's licence and refuses anything
that is not public domain / CC0 / CC BY / MIT-style (attribution is recorded), downloads the latest version as
YOLO, maps its class names to ours (unknown names are dropped), resizes to <= 640 px, and writes:

  OUT/images/rf_<name>_<n>.jpg, OUT/labels/rf_<name>_<n>.txt   (YOLO, our class ids)
  OUT/split.tsv              name<TAB>train|val<TAB>labels  (Roboflow's own valid/test -> val)
  OUT/ATTRIBUTION.md         source, author, licence and URL of every dataset used

The API key is read from a file (or ROBOFLOW_API_KEY) and never written anywhere.
"""
import io, json, os, sys, urllib.parse, urllib.request, zipfile

CLASSES = ['person', 'pet', 'package', 'vehicle']
NAME_MAP = {
    'person': 0, 'people': 0, 'human': 0, 'pedestrian': 0, 'man': 0, 'woman': 0,
    'cat': 1, 'cats': 1, 'dog': 1, 'dogs': 1, 'pet': 1,
    'package': 2, 'packages': 2, 'parcel': 2, 'parcels': 2, 'box': 2, 'boxes': 2, 'cardboard box': 2,
    'small box': 2, 'large box': 2, 'envelope': 2, 'flat envelope': 2, 'delivery': 2, 'carton': 2,
    'car': 3, 'cars': 3, 'truck': 3, 'bus': 3, 'van': 3, 'motorcycle': 3, 'motorbike': 3, 'vehicle': 3,
}
# (short name, workspace, project). Licences are checked live; anything not open is skipped.
DATASETS = [
    ('packages', 'joseph-nelson', 'packages-g0ton'),                 # parcels at house/apartment doors
    ('frontdoor', 'package-detection', 'package-at-front-door'),
    ('parceldetect', 'test-o3fl8', 'parcel-detect-p7xvl'),
    ('pacchi', 'pacchi', 'parcel-ljsjc'),
    ('zouhj', 'zouhj', 'parcel-xwacy'),
    ('pkgveh', 'egge-public', 'packages-vehicles2'),
    ('catocam', 'graham-jones-yynmh', 'catocam'),                    # home CCTV, top-down, IR night
]

MAX_SIDE = 640


def arg(a, k, d):
    return a[a.index(k) + 1] if k in a else d


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def licence_ok(lic):
    """public domain, CC0, plain CC BY or a permissive code licence (MIT/Apache/BSD); never NC (non-commercial), SA (share-alike) or ND."""
    l = ' '.join((lic or '').lower().replace('_', ' ').replace('-', ' ').split())
    if any(t in l.split() for t in ('nc', 'sa', 'nd')) or 'noncommercial' in l or 'non commercial' in l:
        return False
    return ('public domain' in l or 'cc0' in l or l.startswith('cc by')
            or l in ('mit', 'apache 2.0', 'bsd 3 clause'))          # permissive: notice kept in ATTRIBUTION.md


def fetch(short, ws, proj, key, out, rows, attrib):
    from PIL import Image
    q = urllib.parse.urlencode({'api_key': key})
    info = get_json('https://api.roboflow.com/%s/%s?%s' % (ws, proj, q))
    p = info.get('project', {})
    lic = p.get('license') or info.get('license')
    url = 'https://universe.roboflow.com/%s/%s' % (ws, proj)
    if not licence_ok(lic):
        print('  %-12s SKIP: licence %r is not open (%s)' % (short, lic, url)); return 0
    versions = sorted(info.get('versions', []), key=lambda v: int(str(v.get('id', '0')).split('/')[-1]))
    if not versions:
        print('  %-12s SKIP: no versions' % short); return 0
    vid = str(versions[-1]['id']).split('/')[-1]
    exp = get_json('https://api.roboflow.com/%s/%s/%s/yolov8?%s' % (ws, proj, vid, q))
    link = exp.get('export', {}).get('link')
    if not link:
        print('  %-12s SKIP: no export link (%s)' % (short, list(exp)[:5])); return 0
    with urllib.request.urlopen(link, timeout=600) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    import yaml  # noqa: F401  (pyyaml) -- data.yaml holds the class names
    names = yaml.safe_load(z.read('data.yaml'))['names']
    names = names if isinstance(names, list) else [names[k] for k in sorted(names)]
    cmap = {i: NAME_MAP.get(str(n).strip().lower()) for i, n in enumerate(names)}
    n_img = 0
    for f in z.namelist():
        if '/images/' not in f or f.endswith('/'):
            continue
        split = 'train' if f.split('/')[0] == 'train' else 'val'
        lab_name = f.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        lines = []
        if lab_name in z.namelist():
            for l in z.read(lab_name).decode().splitlines():
                v = l.split()
                if len(v) == 5 and cmap.get(int(v[0])) is not None:
                    lines.append('%d %s' % (cmap[int(v[0])], ' '.join(v[1:])))
        im = Image.open(io.BytesIO(z.read(f))).convert('RGB')
        s = MAX_SIDE / max(im.size)
        if s < 1:
            im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BILINEAR)
        name = 'rf_%s_%d' % (short, n_img)
        im.save(os.path.join(out, 'images', name + '.jpg'), quality=90)
        open(os.path.join(out, 'labels', name + '.txt'), 'w').write('\n'.join(lines) + ('\n' if lines else ''))
        rows.append((name, split, '|'.join(lines)))
        n_img += 1
    attrib.append('| %s | [%s/%s](%s) v%s | %s | %d | %s |' % (
        short, ws, proj, url, vid, lic, n_img,
        ', '.join('%s→%s' % (n, CLASSES[c] if c is not None else '–') for n, c in zip(names, cmap.values()))))
    print('  %-12s %s v%s: %d images (%s)' % (short, lic, vid, n_img, attrib[-1].split('|')[-2].strip()))
    return n_img


def main(a):
    out = a[1]
    kf = os.path.expanduser(arg(a, '--key-file', '~/.config/roboflow/key'))
    key = os.environ.get('ROBOFLOW_API_KEY') or (open(kf).read().strip() if os.path.exists(kf) else '')
    if not key:
        raise SystemExit('no Roboflow API key: put it in %s (chmod 600) or ROBOFLOW_API_KEY' % kf)
    only = set(arg(a, '--only', '').split(',')) - {''}
    for d in ('images', 'labels'):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    rows, attrib = [], []
    for short, ws, proj in DATASETS:
        if only and short not in only:
            continue
        try:
            fetch(short, ws, proj, key, out, rows, attrib)
        except Exception as e:  # noqa: BLE001
            print('  %-12s FAILED: %s' % (short, str(e).replace(key, '<key>')[:200]))
    with open(os.path.join(out, 'split.tsv'), 'w') as f:
        f.writelines('%s\t%s\t%s\n' % r for r in rows)
    with open(os.path.join(out, 'ATTRIBUTION.md'), 'w') as f:
        f.write('# Roboflow Universe datasets used\n\n| Name | Source | Licence | Images | Class mapping |\n'
                '|---|---|---|---|---|\n' + '\n'.join(attrib) + '\n')
    print('total %d images from %d datasets' % (len(rows), len(attrib)))


if __name__ == '__main__':
    main(sys.argv)
