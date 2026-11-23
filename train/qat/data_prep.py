#!/usr/bin/env python3
"""How the person / pet / package dataset was selected (kept for reference and re-selection).

build_model.py does not run this: it downloads the exact same image list from
../manifest/*.tsv.gz, so every build trains on identical data. This script needs the full
COCO 2017 annotations (~250 MB) and the Open Images train box CSV (~2.3 GB).

  data_prep.py ROOT [--person N] [--neg N] [--box N] [--jobs 16]

ROOT layout (created): ROOT/coco/annotations (must exist: instances_{train,val}2017.json),
ROOT/oi (classes.csv), and the output ROOT/set/{train,val}/{images,labels}.
Classes: 0 person, 1 pet (cat + dog), 2 package (Open Images "Box").
Labels are YOLO txt (cls cx cy w h, normalised). Images are re-encoded with the
longest side <= 640 px (training crops 16:9 and stretches to 416x320 like raptor).
Downloads only the selected images; re-running skips files already present.
"""
import csv, json, os, random, sys, subprocess, io
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

CLASSES = ['person', 'pet', 'package']
COCO_MAP = {1: 0, 17: 1, 18: 1}                       # person, cat, dog
OI_MAP = {'/m/01g317': 0, '/m/01yrx': 1, '/m/0bt9lr': 1, '/m/025dyy': 2}   # Person, Cat, Dog, Box
OI_BBOX = {'train': 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv',
           'val': 'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv'}
OI_IMG = {'train': 'https://open-images-dataset.s3.amazonaws.com/train/%s.jpg',
          'val': 'https://open-images-dataset.s3.amazonaws.com/validation/%s.jpg'}
COCO_IMG = 'http://images.cocodataset.org/%s2017/%s'
MAX_SIDE = 640


def arg(a, k, d):
    return type(d)(a[a.index(k) + 1]) if k in a else d


def save_item(url, out_img, out_lbl, boxes):
    """boxes: list of (cls, x0, y0, x1, y1) normalised 0..1"""
    from PIL import Image
    if os.path.exists(out_img) and os.path.exists(out_lbl):
        return True
    try:
        data = urlopen(url, timeout=30).read()
        im = Image.open(io.BytesIO(data)).convert('RGB')
    except Exception as e:      # noqa: BLE001
        return False
    s = MAX_SIDE / max(im.size)
    if s < 1:
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BILINEAR)
    tmp_lbl = out_lbl + '.tmp'
    with open(tmp_lbl, 'w') as f:
        for c, x0, y0, x1, y1 in boxes:
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(1, x1), min(1, y1)
            if x1 - x0 > 0.002 and y1 - y0 > 0.002:
                f.write('%d %.6f %.6f %.6f %.6f\n' % (c, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    os.replace(tmp_lbl, out_lbl)
    tmp = out_img + '.tmp'
    im.save(tmp, 'JPEG', quality=90)
    os.replace(tmp, out_img)
    return True


def coco_items(root, split, n_person, n_neg, rng):
    js = json.load(open('%s/coco/annotations/instances_%s2017.json' % (root, split)))
    imgs = {i['id']: i for i in js['images']}
    boxes, has = {}, {}
    for a in js['annotations']:
        if a['category_id'] in COCO_MAP and not a.get('iscrowd'):
            im = imgs[a['image_id']]; x, y, w, h = a['bbox']
            boxes.setdefault(im['id'], []).append((COCO_MAP[a['category_id']], x / im['width'], y / im['height'],
                                                   (x + w) / im['width'], (y + h) / im['height']))
            has.setdefault(im['id'], set()).add(COCO_MAP[a['category_id']])
    crowd = {a['image_id'] for a in js['annotations'] if a.get('iscrowd') and a['category_id'] in COCO_MAP}
    pet = [i for i, s in has.items() if 1 in s and i not in crowd]
    person = [i for i, s in has.items() if s == {0} and i not in crowd]
    neg = [i for i in imgs if i not in has and i not in crowd]
    rng.shuffle(person); rng.shuffle(neg)
    chosen = pet + person[:n_person] + neg[:n_neg]
    return [(COCO_IMG % (split, imgs[i]['file_name']), 'coco_%d' % i, boxes.get(i, [])) for i in chosen]


def oi_items(root, split, n_box, rng):
    path = '%s/oi/%s-bbox.csv' % (root, split)
    if not os.path.exists(path):
        subprocess.run(['curl', '-sSfL', '-o', path + '.part', OI_BBOX[split]], check=True)
        os.replace(path + '.part', path)
    boxes, with_box = {}, set()
    with open(path) as f:
        for r in csv.DictReader(f):
            c = OI_MAP.get(r['LabelName'])
            if c is None or r.get('IsGroupOf') == '1':
                continue
            boxes.setdefault(r['ImageID'], []).append((c, float(r['XMin']), float(r['YMin']), float(r['XMax']), float(r['YMax'])))
            if c == 2:
                with_box.add(r['ImageID'])
    ids = sorted(with_box); rng.shuffle(ids)
    return [(OI_IMG[split] % i, 'oi_%s' % i, boxes[i]) for i in ids[:n_box]]


def main(a):
    root = a[1]
    rng = random.Random(0)
    jobs = arg(a, '--jobs', 16)
    plan = {'train': coco_items(root, 'train', arg(a, '--person', 20000), arg(a, '--neg', 4000), rng)
                     + oi_items(root, 'train', arg(a, '--box', 10000), rng),
            'val': coco_items(root, 'val', 10 ** 9, 500, rng) + oi_items(root, 'val', 2000, rng)}
    for split, items in plan.items():
        d = '%s/set/%s' % (root, split)
        os.makedirs(d + '/images', exist_ok=True); os.makedirs(d + '/labels', exist_ok=True)
        print(split, len(items), 'images to fetch', flush=True)
        ok = 0
        with ThreadPoolExecutor(jobs) as ex:
            futs = [ex.submit(save_item, url, '%s/images/%s.jpg' % (d, n), '%s/labels/%s.txt' % (d, n), b)
                    for url, n, b in items]
            for k, f in enumerate(futs):
                ok += f.result()
                if k % 2000 == 0:
                    print('  %s %d/%d ok %d' % (split, k, len(items), ok), flush=True)
        print(split, 'done', ok, 'of', len(items), flush=True)


if __name__ == '__main__':
    main(sys.argv)
