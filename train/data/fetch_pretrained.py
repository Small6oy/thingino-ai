#!/usr/bin/env python3
"""Fetch the openly licensed pretrained weights our architectures can start from.

  python3 data/fetch_pretrained.py OUT [--only NAME,...]

Every file is pinned by sha256: a download that differs is refused and deleted. Each source's
licence is fetched next to its weights, so OUT can be shipped as it is. Nothing here comes from
Ingenic: a model trained from these starts is ours to publish (check the dataset terms, see
docs/pretrained.md).

  OUT/<name>/<file>    the weights
  OUT/<name>/LICENSE   the upstream licence (or model card, for Hugging Face)
  OUT/SOURCES.md       source, licence and URL of every file fetched

.pth / .ckpt files are Python pickles: load them only with torch.load(..., weights_only=True).
"""
import hashlib, os, sys, urllib.request

YOLOX = 'https://github.com/Megvii-BaseDetection/YOLOX'
NANODET = 'https://github.com/RangiLyu/nanodet'
HF = 'https://huggingface.co/timm/'

# name: (what, licence, licence URL, [(file, URL, sha256)])
SOURCES = {
    'yolox_nano': ('YOLOX-Nano, COCO detector (Megvii, release 0.1.1rc0; 0.91M params, 416 px)', 'Apache-2.0',
                   'https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/LICENSE',
                   [('yolox_nano.pth', YOLOX + '/releases/download/0.1.1rc0/yolox_nano.pth',
                     'cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b')]),
    'yolox_tiny': ('YOLOX-Tiny, COCO detector (Megvii, release 0.1.1rc0; 5.06M params, 416 px)', 'Apache-2.0',
                   'https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/LICENSE',
                   [('yolox_tiny.pth', YOLOX + '/releases/download/0.1.1rc0/yolox_tiny.pth',
                     '9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0')]),
    'mnv4_small_050': ('MobileNetV4 conv-small 0.5x, ImageNet-1k classifier (timm, Ross Wightman)', 'Apache-2.0',
                       HF + 'mobilenetv4_conv_small_050.e3000_r224_in1k/resolve/main/README.md',
                       [('model.safetensors', HF + 'mobilenetv4_conv_small_050.e3000_r224_in1k/resolve/main/model.safetensors',
                         'f99769e8f0623ac941ce18d53bbc54b5f38d499af91034e290501bf78efb733d')]),
    'mnv4_small': ('MobileNetV4 conv-small, ImageNet-1k classifier (timm, Ross Wightman)', 'Apache-2.0',
                   HF + 'mobilenetv4_conv_small.e1200_r224_in1k/resolve/main/README.md',
                   [('model.safetensors', HF + 'mobilenetv4_conv_small.e1200_r224_in1k/resolve/main/model.safetensors',
                     '5a2ef04d419ce6d1bf27bfa735bb200d3f8d8997c3ac36320f5bf30382f6b43c')]),
    'nanodet_plus_m_416': ('NanoDet-Plus-m 416, COCO detector (RangiLyu, v1.0.0-alpha-1; ShuffleNetV2 1.0x)', 'Apache-2.0',
                           'https://raw.githubusercontent.com/RangiLyu/nanodet/main/LICENSE',
                           [('nanodet-plus-m_416_checkpoint.ckpt', NANODET + '/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416_checkpoint.ckpt',
                             'dbfdf81f05e90f7d84f6171cf82b87f68cec0a2532543fd0d9da5b448169c385')]),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def get(url, path):
    tmp = path + '.part'
    req = urllib.request.Request(url, headers={'User-Agent': 'thingino-ai-fetch'})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, 'wb') as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(tmp, path)


def main(argv):
    if not argv or argv[0].startswith('-'):
        raise SystemExit(__doc__)
    out = argv[0]
    only = None
    if '--only' in argv:
        only = set(argv[argv.index('--only') + 1].split(','))
        bad = only - set(SOURCES)
        if bad:
            raise SystemExit('unknown: %s (have %s)' % (', '.join(sorted(bad)), ', '.join(SOURCES)))
    os.makedirs(out, exist_ok=True)
    lines = ['# Pretrained weights', '', '| Name | What | Licence | Source |', '|---|---|---|---|']
    for name, (what, lic, lic_url, files) in SOURCES.items():
        if only and name not in only:
            continue
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        for fn, url, want in files:
            p = os.path.join(d, fn)
            if not (os.path.exists(p) and sha256(p) == want):
                print('%s: fetching %s' % (name, fn), flush=True)
                get(url, p)
            have = sha256(p)
            if have != want:
                os.remove(p)
                raise SystemExit('%s: %s sha256 %s, expected %s (removed)' % (name, fn, have, want))
            lines.append('| %s | %s | %s | %s |' % (name, what, lic, url))
        lp = os.path.join(d, 'LICENSE')
        if not os.path.exists(lp):
            get(lic_url, lp)
        print('%s: ok (%s)' % (name, lic))
    with open(os.path.join(out, 'SOURCES.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:])
