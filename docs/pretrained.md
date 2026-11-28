# Open pretrained starting points

All model types side by side, with runtime support: [model-types.md](model-types.md).

Ingenic-based starts (`--init persondet`, and `--arch yolon`) give models we can't publish. These
starts are openly licensed, so a model trained from them (on open data) can ship as a Thingino default.

Fetch them with `python3 train/data/fetch_pretrained.py OUT`. Every file is pinned by sha256, and the
upstream licence is saved next to it.

| Name | What | Licence | Status |
|---|---|---|---|
| `yolox_nano` | YOLOX-Nano COCO detector (Megvii, release 0.1.1rc0) | Apache-2.0 | **wired in:** `--arch yoloxn --init yolox` |
| `yolox_tiny` | YOLOX-Tiny COCO detector (5.06M params) | Apache-2.0 | fetched only (a wider T41 variant) |
| `mnv4_small_050` | MobileNetV4 conv-small 0.5x, ImageNet-1k (timm) | Apache-2.0 | **wired in:** `--arch mnv4n --init mnv4` (+ YOLOX-Nano neck/heads) |
| `mnv4_small` | MobileNetV4 conv-small 1x | Apache-2.0 | fetched only |
| `nanodet_plus_m_416` | NanoDet-Plus-m COCO detector (ShuffleNetV2) | Apache-2.0 | fetched only: no T31 port (see below) |

The weights carry their own licence, but they were trained on COCO or ImageNet, which have their own
terms. Check those before shipping a model as a default.

## YOLOX-Nano on the T31 (`train/qat/yoloxn.py`)

The graph keeps YOLOX-Nano's backbone and neck, with these T31 changes:

- **ReLU** instead of SiLU: the T31 has unsigned activation codes only.
- **Stem:** Focus + 3x3 is exactly a 6x6 stride-2 conv (`stem='focus6'`). On the T31 that costs ~57M
  MACs on the slow type 46 path, and the int16 accumulator bound leaves ~3-bit weights. The default is a
  3x3 stride-2 stem (`conv3`, ~14M MACs), started from the 6x6's centre taps.
- **No SPP pools:** the T31 has no stride-1 max pool, so the 5/9/13 pools become identity (conv2's
  four input groups are summed).
- **Stride-2 depthwise padding** is bottom/right only (as Ingenic's graphs), not YOLOX's 1 on every
  side, so pretrained stride-2 layers see a one-pixel shift until fine-tuned.
- **Heads:** ours, YOLOv5-style anchors on strides 8/16/32, so raptor-hal's decode is unchanged. They
  are fed by YOLOX's cls and reg branches. Objectness and class rows start from YOLOX's predictions
  (vehicle = car, motorcycle, bus, truck; pet = cat, dog); box rows start at the anchors.
- **Residual adds:** libjzdl's type 78 stores the mean of its inputs. The loader folds the factor 2
  into the following layers, so the float graph equals YOLOX (with the changes above) exactly:
  `python3 train/qat/yoloxn.py WEIGHTS --check`.

Input: Megvii's checkpoints take BGR pixels 0..255 without normalisation (checked: person recall
0.82 on our validation images, against 0.01 for ImageNet-normalised RGB). The first conv absorbs it.

Training: a float stage (default 80 epochs since round 2, BatchNorm on) re-learns around the changes,
then BatchNorm folding, activation calibration and quantisation-aware training, as `--init scratch`
does. Adding a bigger teacher with `--distill yolox_s` or `yolox_m` ([distillation.md](distillation.md))
lifts 3-class T31 exact mean AP50 from ~0.46 to ~0.58.

Cost: ~341M MACs per 416x320 frame with the 3x3 stem, most of it in 1x1 convs (302M).

## MobileNetV4 backbone on the T31 (`train/qat/mnv4.py`, `--arch mnv4n`)

MobileNetV4-conv-small-0.5 (timm) as the backbone, then three new 1x1 adapters (random start) into
YOLOX-Nano's neck and heads (Megvii's weights). Features: stride 8 (32 ch), 16 (48 ch), 32 (480 ch).

- **5x5 depthwise** kernels become their centre 3x3 (the T31 has 3x3 depthwise only).
- **Linear layers** (UIR `dw_start`, `pw_proj`, and so the residual stream) can be negative; T31
  activations are unsigned. They store value + a per-channel offset, measured on 32 training images
  with the T31's padding (stored 0 = true -offset at the borders), with a 1.5x margin. The next
  layer's bias takes the offset back out (`train/qat/mapweights.py`).
- **Re-centring:** offsets would add up along the residual stream (to hundreds), leaving its 5-bit
  codes almost all offset and breaking the requant limit downstream. Each residual add is followed by
  a 3x3 depthwise layer with only its centre tap (a per-channel scale + bias, ~0.2M MACs a block) that
  resets the stream to its own offset. Clipping the stream at 0 instead loses the pretraining
  (feature cosine 0.07 at stride 32); with re-centring it is 0.98-1.00 on validation images.
- One channel of block 2.5 reaches about -860 (median offsets ~13). It sets that tensor's scale;
  the float stage and QAT have to tame it.
- Check: `python3 train/qat/mnv4.py MNV4.safetensors YOLOX.pth` (float graph == MobileNetV4 with the
  changes above, worst error 4e-6). The T31 export is bit-exact.

**Cost:** ~380M MACs per frame, of which 81.5M are dense 3x3 convs (stem, blocks 0.0 and 1.0) on the
T31's slow type 46 path (~0.6 GMAC/s, so ~135 ms for those alone). Expect it to be slower than YOLO-N
on the T31; it suits the T41's NNA better.

## NanoDet-Plus: no T31 port

ShuffleNetV2 splits each tensor into two channel halves, works on one, then concatenates and
shuffles. The T31's graph has no channel slice: a split (type 33) copies the whole tensor. The halves
can be emulated exactly with 1x1 convs (0/1 weights for the selection, zero weights for the unused
half), but that adds about C^2 x 3/4 MACs per unit: ~84M (stage 2) + ~168M (stage 3) + ~84M (stage 4),
~340M MACs on top of the network itself, which is more than YOLOX-Nano costs in total. It also uses
LeakyReLU (needs a swap) and a GFL head (distribution box decode, not ours). Its weights stay fetched
for a possible T41-only model later.

## Before training: what the starts score (`train/eval_start.py`)

`THINGINO_AI_CLASSES=person,pet,vehicle python3 train/eval_start.py WORK/dataset` scores each start on
the validation set with the training code's own AP@0.5, step by step. v2 validation set, 6,223 images,
classes person / pet / vehicle:

| Variant | person | pet | vehicle | mean |
|---|---|---|---|---|
| YOLOX-Nano as released (SiLU, SPP, its own head and decode) | 0.657 | 0.662 | 0.556 | **0.625** |
| + the T31 graph changes (ReLU, no SPP, bottom/right stride-2 pads, 3x3 stem), its own head | 0.000 | 0.000 | 0.000 | 0.000 |
| `yoloxn` float start / BatchNorm re-measured / int8 QAT start | 0.000 | 0.001 | 0.000 | ~0 |
| `mnv4n` float start / BatchNorm re-measured | 0.000 | 0.000 | 0.000 | 0 |

On 64 images, each T31 change alone: bottom/right pads 0.28 mean, no SPP 0.05, 3x3 stem 0.11, ReLU 0.00.
So the starts carry YOLOX's features, but on the T31 graph they only work after the float fine-tune.
An untrained `.bin` on a camera is good for timing only. For comparison, the trained models on the same
three classes: YOLO-N 0.465, persondet v2 0.421 (mean AP50, full validation set).

## Three classes: person, pet, vehicle

`build_model.py --classes person,pet,vehicle` trains a subset of the manifest's classes, in that order.
Boxes of the other classes are dropped when labels load (their images stay, as background). For a GPU
machine, `scripts/train-open-3class.sh [yoloxn|mnv4n|both]` runs the quick smoke test, then the full
build, for the open starts with these three classes. Results: `train/out/open3-<arch>/`.

## Input size: 384x224 for the T31 (`--input WxH`, yoloxn / mnv4n)

The camera's detection frame is 640x360 (16:9). At 416x320 every frame is stretched vertically; at
384x224 or 448x256 it keeps its shape. YOLOX-Nano as released, 1,500 validation images, mean AP50
(person / pet / vehicle):

| Input | person | pet | vehicle | mean | compute | yoloxn on the v3 (ojzdl) |
|---|---|---|---|---|---|---|
| 416x320 (stretched) | 0.652 | 0.644 | 0.586 | 0.627 | 100 % | 126 ms standalone, 131 ms in rvd |
| **384x224** (16:9) | 0.645 | 0.679 | 0.593 | **0.639** | 65 % | **75 ms** standalone, 85 ms in rvd |
| 448x256 (16:9) | 0.658 | 0.706 | 0.594 | 0.652 | 86 % | not timed |

384x224 is faster than persondet v2 (93 ms in rvd) and, before training, at least as accurate as
416x320. rvd scales 640x360 to the model's input itself (`640x360 -> 384x224`). On a GPU machine:
`INPUT=384x224 scripts/train-open-3class.sh yoloxn` (run name `open3-yoloxn-384x224`).

## Requant guard

`calib.fit_requant()` runs after activation calibration (QAT start and self-test). Where a layer's
weight step would exceed its output scale (libjzdl's requant shift would go negative and the export
fails; new random layers can start there), it raises that output scale just enough.
