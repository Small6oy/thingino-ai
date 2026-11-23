#!/usr/bin/env python3
"""Build train/manifest/ (the dataset build_model.py downloads) from COCO 2017 and Open Images.

  python3 data/build_manifest.py SRC [--out manifest] [--seed 0]

SRC holds the public annotation files (downloaded once, ~3 GB):
  SRC/coco/annotations/instances_{train,val}2017.json   (annotations_trainval2017.zip)
  SRC/oi/{train,val}-bbox.csv                             (oidv6-train / v5 validation bbox csv)

Output: manifest/{train,val}.tsv.gz, one image per line: `name<TAB>cls cx cy w h|...`
(YOLO, normalised to the full image), and manifest/classes.json. No images are downloaded here.

Every chosen image is labelled for ALL classes its source annotates, so no person, pet, package
or vehicle sits in an image unlabelled (an unlabelled object teaches the model "background").
"""
import csv, gzip, json, os, random, sys
from collections import defaultdict

CLASSES = ['person', 'pet', 'package', 'vehicle']
PERSON, PET, PACKAGE, VEHICLE = range(4)
COCO_MAP = {1: PERSON, 17: PET, 18: PET,                               # person, cat, dog
            3: VEHICLE, 4: VEHICLE, 6: VEHICLE, 8: VEHICLE}            # car, motorcycle, bus, truck
OI_MAP = {'/m/01g317': PERSON, '/m/01yrx': PET, '/m/0bt9lr': PET, '/m/025dyy': PACKAGE,
          '/m/0k4j': VEHICLE, '/m/07r04': VEHICLE, '/m/01bjv': VEHICLE, '/m/0h2r6': VEHICLE,
          '/m/04_sv': VEHICLE}                                         # Car, Truck, Bus, Van, Motorcycle
# how many images to take per group (train); val takes all of COCO val that has a class + a sample of OI
QUOTA = dict(coco_pet=10 ** 9, coco_vehicle=14000, coco_person=15000, coco_neg=4000,
             oi_package=10 ** 9, oi_pet=12000, oi_vehicle=6000)
VAL_QUOTA = dict(coco_neg=500, oi_package=10 ** 9, oi_pet=600, oi_vehicle=400)


def arg(a, k, d):
    return type(d)(a[a.index(k) + 1]) if k in a else d


def fmt(boxes):
    out = []
    for c, x0, y0, x1, y1 in boxes:
        x0, y0, x1, y1 = max(0., x0), max(0., y0), min(1., x1), min(1., y1)
        if x1 - x0 > 0.002 and y1 - y0 > 0.002:
            out.append('%d %.5f %.5f %.5f %.5f' % (c, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    return '|'.join(out)


def coco(src, split, quota, rng):
    js = json.load(open('%s/coco/annotations/instances_%s2017.json' % (src, split)))
    imgs = {i['id']: i for i in js['images']}
    boxes, has, crowd = defaultdict(list), defaultdict(set), set()
    for a in js['annotations']:
        c = COCO_MAP.get(a['category_id'])
        if c is None:
            continue
        if a.get('iscrowd'):
            crowd.add(a['image_id']); continue          # a crowd region can't be labelled box by box
        im = imgs[a['image_id']]; x, y, w, h = a['bbox']
        boxes[im['id']].append((c, x / im['width'], y / im['height'], (x + w) / im['width'], (y + h) / im['height']))
        has[im['id']].add(c)
    ok = [i for i in has if i not in crowd]
    groups = dict(pet=[i for i in ok if PET in has[i]],
                  vehicle=[i for i in ok if VEHICLE in has[i] and PET not in has[i]],
                  person=[i for i in ok if has[i] == {PERSON}],
                  neg=[i for i in imgs if i not in has and i not in crowd])
    chosen = []
    for g, ids in groups.items():
        ids = sorted(ids); rng.shuffle(ids)
        chosen += ids[:quota.get('coco_' + g, 10 ** 9)]
    return [('coco_%d' % i, fmt(boxes.get(i, []))) for i in sorted(set(chosen))]


def oi(src, split, quota, rng):
    boxes, has, bad = defaultdict(list), defaultdict(set), set()
    with open('%s/oi/%s-bbox.csv' % (src, split)) as f:
        for r in csv.DictReader(f):
            c = OI_MAP.get(r['LabelName'])
            if c is None:
                continue
            if r['IsGroupOf'] == '1' or r['IsDepiction'] == '1':
                bad.add(r['ImageID']); continue          # group boxes / drawings: skip the whole image
            boxes[r['ImageID']].append((c, float(r['XMin']), float(r['YMin']), float(r['XMax']), float(r['YMax'])))
            has[r['ImageID']].add(c)
    ok = [i for i in has if i not in bad]
    groups = dict(package=[i for i in ok if PACKAGE in has[i]],
                  pet=[i for i in ok if PET in has[i] and PACKAGE not in has[i]],
                  vehicle=[i for i in ok if VEHICLE in has[i] and not has[i] & {PET, PACKAGE}])
    chosen = []
    for g, ids in groups.items():
        ids = sorted(ids); rng.shuffle(ids)
        chosen += ids[:quota.get('oi_' + g, 0)]
    return [('oi_%s' % i, fmt(boxes[i])) for i in sorted(set(chosen))]


def main(a):
    src = a[1]
    out = arg(a, '--out', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'manifest'))
    rng = random.Random(arg(a, '--seed', 0))
    os.makedirs(out, exist_ok=True)
    stats = {}
    for split, q in (('train', QUOTA), ('val', VAL_QUOTA)):
        rows = coco(src, split, q, rng) + oi(src, split, q, rng)
        with gzip.GzipFile(os.path.join(out, split + '.tsv.gz'), 'wb', mtime=0) as f:
            f.write(''.join('%s\t%s\n' % r for r in rows).encode())
        n = [0] * len(CLASSES); imgs = [0] * len(CLASSES)
        for _, lab in rows:
            seen = set()
            for b in lab.split('|') if lab else []:
                c = int(b.split()[0]); n[c] += 1; seen.add(c)
            for c in seen:
                imgs[c] += 1
        stats[split] = dict(images=len(rows), boxes=dict(zip(CLASSES, n)), images_with=dict(zip(CLASSES, imgs)))
        print(split, json.dumps(stats[split]))
    json.dump(dict(version=2, classes=CLASSES,
                   class_sources={'person': 'COCO person, OI Person', 'pet': 'COCO cat+dog, OI Cat+Dog',
                                  'package': 'OI Box', 'vehicle': 'COCO car/truck/bus/motorcycle, OI Car/Truck/Bus/Van/Motorcycle'},
                   stats=stats), open(os.path.join(out, 'classes.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv)
