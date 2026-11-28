# Distillation: a small camera model learning from a bigger one

`build_model.py --distill T` trains the camera model (the *student*) against the labels **and**
against a *teacher*'s outputs. The teacher never ships; it exists only during training.

## Why

The T31 model can't have what makes YOLOX accurate: SiLU (the T31 needs ReLU), the SPP pools, the
Focus stem. Measured on our validation set (AP50, 3 classes person / pet / vehicle):

| | person | pet | vehicle |
|---|---|---|---|
| YOLOX-S as released (COCO, no training on our data; 300 val images) | 0.709 | 0.704 | 0.666 |
| YOLOX-Nano as released (same 300 images) | 0.654 | 0.622 | 0.570 |
| our trained yoloxn, T31 exact (full val, 6,223 images) | 0.407 | 0.629 | 0.342 |

The training logs also show every run losing accuracy when quantisation switches on (float end vs
first QAT epoch) that 30 QAT epochs never fully win back: -0.014 to -0.073 mean AP50.

A teacher's soft outputs ("0.7 person, 0.2 pet", objectness on every cell, objects the labels
missed) carry more than the labels do, so the student gets more signal per image.

## Teachers

| `--distill` | Teacher | Used in | Notes |
|---|---|---|---|
| `none` (default) | - | - | as before |
| `float` | the run's own float network | QAT (phase 2) | cheapest: targets the quantisation loss only |
| `yolox_tiny`, `yolox_s`, `yolox_m`, `yolox_l` | YOLOX as Megvii released it (SiLU, SPP, Focus; Apache-2.0) with our heads | float stage + QAT | `--arch yoloxn` / `mnv4n` only. A teacher stage (phase 0) first fine-tunes it on our classes in float |

The teacher's heads are ours (YOLOv5-style anchors on strides 8 / 16 / 32, started from YOLOX's
own objectness / class predictions as `yoloxn.from_yolox` does), so student and teacher are
compared cell by cell (`train.kd_loss`): objectness BCE against the teacher's probability on every
cell, class BCE against its class probabilities, and a box term, both weighted by its objectness.
`--kd W` scales the distillation loss (default 1.0). Code: `train/qat/teacher.py`, `train/qat/train.py`
(`kd_loss`, `teacher_outs`); check: `python3 qat/teacher.py --check` (YOLOX-Nano teacher features ==
`yoloxn.yolox_forward`, max diff 0).

Licence: every teacher is Apache-2.0 and trained on the same open data, so a distilled model stays
publishable exactly like the plain one.

## Size, GPU memory and speed

| Teacher | Params | GMACs at 416x320 (forward) | Phase 0 peak VRAM (`bs=64`, bf16) | Phase 0 peak VRAM (`bs=32`, bf16) |
|---|---:|---:|---:|---:|
| `yolox_tiny` | 5.05M | 2.46 | — | — |
| `yolox_s` | 8.95M | 4.32 | 4.6 GB alloc / 5.1 GB res | 2.4 GB alloc / 2.8 GB res |
| `yolox_m` | 25.3M | 11.9 | 8.2 GB alloc / 9.2 GB res | 4.4 GB alloc / 4.9 GB res |
| `yolox_l` | 54.2M | 25.2 | — | — |
| (student `yoloxn`) | 0.90M | 0.34 | 2.6–2.9 GB (Phase 1) / 7.2–7.3 GB (Phase 2 QAT) | — |

Measured on an NVIDIA RTX Spark N1X (`sm_121`, CUDA 13.4, PyTorch 2.14, bf16, `bs=64`, with PR #83's
compiled `channels_last` frozen teacher). Start with `--quick` (~1–2 min on a Blackwell-class GPU);
on an out-of-memory error lower `--teacher-bs`:

- **16 GB cards without bf16 (V100):** `yolox_s`, `--teacher-bs 32`. The teacher stage trains in fp32
  there (bf16 autocast needs compute capability 8.0+).
- **Ampere and newer, 12 GB+ or unified memory:** both `yolox_s` (~5.1 GB reserved) and `yolox_m`
  (~9.2 GB reserved) fit comfortably at `--bs 64` in bf16; `yolox_l` needs `--teacher-bs 32` or more
  VRAM.
- **Measured full-run wall times (RTX Spark N1X, `bs=64`, 69,854 train images):**
  - `DISTILL=yolox_s`: **`8h 05m`** total (`1m 09s` `--quick`) — Phase 0 ~312 img/s (30 ep), Phase 1 ~416 img/s (80 ep), Phase 2 QAT ~233 img/s (30 ep).
  - `DISTILL=yolox_m`: **`11h 22m`** total (`1m 23s` `--quick`) — Phase 0 ~164 img/s (30 ep), Phase 1 ~325 img/s (80 ep), Phase 2 QAT ~198 img/s (30 ep).

## Round 2 results (`yoloxn`, 3 classes: person / pet / vehicle)

All runs below are 100 % open Apache-2.0 (`scripts/train-open-3class.sh yoloxn`), bit-exact T31
exports (`max_diff = 0`). Training-time AP50 is on the 3,000-image validation split (`train.py`);
full-val (6,223 images) is listed where published.

| Run | Teacher | Schedule (teacher / float / QAT) | Teacher AP50 | Float AP50 | T31 exact AP50 (`person / pet / vehicle`, **mean**) | T31 `≥ 32 px` mean | Wall time (Spark N1X) |
|---|---|---|---:|---:|---|---:|---:|
| Round 1 baseline (`open3-yoloxn`) | none | 0 / 20 / 30 ep | — | — | `0.420 / 0.621 / 0.349` (**0.463**) | 0.548 | 2h 51m |
| Round 2 Run C (`open3-yoloxn-kd-yolox_m-r2`) | `yolox_m` (25.3M) | 30 / 80 / 30 ep | 0.680 | 0.615 | `0.532 / 0.732 / 0.472` (**0.579**; full val **0.578**) | 0.678 | 11h 22m |
| Round 2 Run D (`open3-yoloxn-kd-yolox_s-r2`) | `yolox_s` (8.95M) | 30 / 80 / 30 ep | 0.668 | 0.612 | `0.528 / 0.748 / 0.478` (**0.585**) | 0.684 | 8h 05m |

- Both distilled runs gained **+0.116 to +0.122 mean AP50** (+0.130 to +0.136 on objects `≥ 32 px`)
  over the Round 1 baseline on the exact same 0.90M-parameter T31 `yoloxn` student.
- Run C (`yolox_m` teacher) is published upstream as
  [`model-yoloxn3-v1`](https://github.com/Small6oy/thingino-ai/releases/tag/model-yoloxn3-v1)
  (full-val 6,223 images: `0.523 / 0.751 / 0.460`, mean `0.578`; T41 MSE int8 holds on 1,500 frames:
  `0.537 / 0.711 / 0.477`).
- `yolox_s` (8.95M params) trained **3h 17m faster** and used ~45 % less Phase 0 VRAM than `yolox_m`
  while matching/slightly edging out `yolox_m` on the 3,000-image val split (`0.585` vs `0.579`),
  making `DISTILL=yolox_s` a strong default teacher for `yoloxn`.

## Commands

```sh
# round 2 baseline: no distillation, 80 float epochs (the new default for --init yolox / mnv4)
scripts/train-open-3class.sh yoloxn

# QAT self-distillation (no extra download)
DISTILL=float scripts/train-open-3class.sh yoloxn

# bigger teacher
DISTILL=yolox_s scripts/train-open-3class.sh yoloxn --teacher-bs 32       # 16 GB cards
DISTILL=yolox_m scripts/train-open-3class.sh yoloxn                       # 24 GB+ / unified memory
```

Run names get `-kd-<teacher>` and a round tag (`TAG`, default `r2`), so a new round never resumes an
older run. Send back `train/out/<run>/` as usual; `report.json` records `distill`, `kd` and
`teacher_epochs`, and `log.txt` has the teacher stage's own AP50 (`teacher epoch ...` lines).
