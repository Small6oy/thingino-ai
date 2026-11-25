# Model types

The detector graphs we train (`train/build_model.py --arch ...`), how they differ, and what runs them.
All are single-pass detectors: a backbone that shrinks the image, a short neck that mixes scales, and
heads that predict boxes on a grid. They differ in their building blocks and in how many scales they
look at. That decides both accuracy and which runtimes can run them. Input: 416x320 RGB.

## Architectures

| | persondet | YOLO-N | YOLOX-Nano | MobileNetV4 + YOLOX neck |
|---|---|---|---|---|
| `--arch` | `persondet` | `yolon` | `yoloxn` | `mnv4n` |
| Origin | Ingenic's person detector | Ingenic's YOLOv5 sample | Megvii YOLOX-Nano | timm MobileNetV4-conv-small-0.5 + Megvii YOLOX-Nano neck/heads |
| Starting weights (`--init`) | Ingenic (`persondet`) or random (`scratch`) | Ingenic (`persondet`) | Megvii COCO (`yolox`) or `scratch` | timm ImageNet + Megvii COCO (`mnv4`) or `scratch` |
| Weights licence | Ingenic: **private** (scratch: ours) | Ingenic: **private** | Apache-2.0: **publishable** | Apache-2.0: **publishable** |
| Building blocks | 3x3 stride-2 image conv, 3x3 depthwise, 1x1, 2x2 max-pool, zero-insert upsample, concat | as persondet, plus **residual adds**, nearest upsample, stride-2 depthwise | as YOLO-N (CSP blocks), SPP pools removed | as YOLOX-Nano, plus **dense 3x3 convs inside the graph**, linear bottlenecks with offsets, a re-centring layer after each residual |
| Heads (strides) | 2 (16, 32) | 3 (8, 16, 32) | 3 (8, 16, 32) | 3 (8, 16, 32) |
| Multiply-adds per frame | 202M | 483M | 341M | 380M (81.5M on the T31's slow dense-conv path) |
| T31 (Wyze Cam v3) speed | ~104 ms, +7 % CPU at 1 Hz (measured) | ~200 ms, +15 % CPU (measured) | ~150 ms (estimate) | likely slower than YOLO-N (estimate) |
| Strength | cheapest; proven live | small and distant objects (stride-8 head) | stride-8 head at ~70 % of YOLO-N's cost; publishable | strong ImageNet backbone; publishable |
| Weak spot | small objects (no stride-8 head: small-person AP50 0.07) | twice persondet's CPU | not trained yet | slow on the T31; outlier channel in block 2.5; not trained yet |
| Best result so far (AP50, full data-v2 val, 6,223 frames) | v2: mean 0.463 (person .440, pet .556, package .592, vehicle .266) | 4-class helper build: mean 0.490 (.477 / .558 / .564 / .360) | waiting for a training run | waiting for a training run |
| Code | `train/qat/net.py`, `skeleton.py` | `train/qat/graphnet.py` (Ingenic's graph) | `train/qat/yoloxn.py` | `train/qat/mnv4.py` |

Not in the list:
- **NanoDet-Plus** (Apache-2.0, weights fetched): its ShuffleNet channel split and shuffle have no T31
  layer; emulating them adds ~340M multiply-adds per frame. A candidate for a T41-only model.
- **TinyDet:** removed (too slow on the T31: 317-834 ms; weights' terms unconfirmed).

Details of the open-weight starts (what was changed for the T31 and why): [pretrained.md](pretrained.md).

## Runtime support

T31 (Wyze Cam v3) runs JZDL models on the CPU: Ingenic's `libjzdl` or our open `ojzdl`
(`tools/ojzdl`). T41 (Wyze Cam v4) runs NNA models: Ingenic's converter (magik) makes the `.bin`,
run by Ingenic's `venus` or our open `onna` (`tools/onna`).

| | persondet | YOLO-N | YOLOX-Nano | MobileNetV4 + YOLOX neck |
|---|---|---|---|---|
| T31 export == our integer reference | yes | yes | yes | yes |
| T31 `libjzdl` (bit-exact gate) | yes | yes | yes | yes (since PR #43) |
| T31 `ojzdl` (bit-exact gate) | yes | yes | yes | yes (since PR #43) |
| T41 export converts (magik) | yes | yes | yes | yes |
| T41 `onna` (host: byte-identical to Ingenic's PC executor) | yes | yes | yes | yes |

Status 2026-11-25. "Bit-exact gate": `tools/ojzdl/gate.sh` runs Ingenic's libjzdl (QEMU), ojzdl (MIPS under QEMU and
native) and our integer reference (`tools/jzdl/jzdl_ref.py`) on six frames; every output must be
identical. onna results are on the host (CPU engine and the NNA oracle, heads byte-identical to the PC executor
on one frame); not yet on a v4. Stride-2 and x64-requant convs run on the CPU for now.

PR #43 fixed two mismatches found with mnv4n: libjzdl takes the int16 requant shift modulo 16
(the exporter now caps it at 15), and ojzdl's dense 3x3 conv inside the graph used the image's
padding. See [jzdl-format.md](jzdl-format.md) "Arithmetic".
