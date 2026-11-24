#!/usr/bin/env python3
"""Fill in classes a source dataset doesn't annotate, with a big COCO detector (pseudo-labels).

  python3 data/fill_labels.py DIR [--conf 0.7] [--device mps|cuda|cpu]

DIR is a fetch_roboflow.py output (images/, labels/, split.tsv). Most of those datasets label one
class only (e.g. parcels), so a person or car standing next to the parcel would be "background"
for our model. For each image, every class its source does NOT annotate (SOURCE_CLASSES) is filled
from torchvision's Faster R-CNN (ResNet-50 FPN v2, COCO, BSD-3) at a high score threshold. Boxes for
classes the source does annotate are never touched. Rewrites labels/*.txt and split.tsv in place and
records what it added in fill_report.json.
"""
import json, os, sys

# our classes a dataset annotates itself (prefix of the image name after rf_)
SOURCE_CLASSES = {'packages': {2}, 'frontdoor': {2}, 'parceldetect': {2}, 'pacchi': {2}, 'zouhj': {2},
                  'pkgveh': {2, 3}, 'catocam': {0, 1}}
# COCO category ids (torchvision numbering) -> our class
COCO_TO_OURS = {1: 0, 17: 1, 18: 1, 3: 3, 4: 3, 6: 3, 8: 3}   # person, cat, dog, car, motorcycle, bus, truck


def arg(a, k, d):
    return type(d)(a[a.index(k) + 1]) if k in a else d


def main(a):
    import torch, torchvision
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    d = a[1]
    conf = arg(a, '--conf', 0.7)
    dev = arg(a, '--device', 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    w = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=w, box_score_thresh=conf).eval().to(dev)
    rows = [l.rstrip('\n').split('\t') for l in open(os.path.join(d, 'split.tsv'))]
    added, out = {}, []
    with torch.no_grad():
        for k, (name, split, lab) in enumerate(rows):
            src = name.split('_')[1]
            fill = {0, 1, 3} - SOURCE_CLASSES.get(src, {0, 1, 2, 3})
            boxes = [b for b in lab.split('|') if b]
            if fill:
                im = Image.open(os.path.join(d, 'images', name + '.jpg')).convert('RGB')
                W, H = im.size
                r = model([to_tensor(im).to(dev)])[0]
                for (x0, y0, x1, y1), c in zip(r['boxes'].tolist(), r['labels'].tolist()):
                    ours = COCO_TO_OURS.get(c)
                    if ours in fill:
                        boxes.append('%d %.5f %.5f %.5f %.5f' % (ours, (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H))
                        added[ours] = added.get(ours, 0) + 1
            lab = '|'.join(boxes)
            open(os.path.join(d, 'labels', name + '.txt'), 'w').write(lab.replace('|', '\n') + ('\n' if lab else ''))
            out.append((name, split, lab))
            if k % 500 == 0:
                print('%d/%d  added so far %s' % (k, len(rows), added), flush=True)
    with open(os.path.join(d, 'split.tsv'), 'w') as f:
        f.writelines('%s\t%s\t%s\n' % r for r in out)
    rep = dict(model='torchvision fasterrcnn_resnet50_fpn_v2 COCO_V1', score_threshold=conf,
               source_classes={k: sorted(v) for k, v in SOURCE_CLASSES.items()},
               added_boxes={['person', 'pet', 'package', 'vehicle'][c]: n for c, n in sorted(added.items())})
    json.dump(rep, open(os.path.join(d, 'fill_report.json'), 'w'), indent=1)
    print(json.dumps(rep['added_boxes']))


if __name__ == '__main__':
    main(sys.argv)
