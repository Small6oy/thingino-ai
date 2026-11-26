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

## Size and GPU memory

| Teacher | Params | GMACs at 416x320 (forward) |
|---|---|---|
| yolox_tiny | 5.05M | 2.46 |
| yolox_s | 8.95M | 4.32 |
| yolox_m | 25.3M | 11.9 |
| yolox_l | 54.2M | 25.2 |
| (student yoloxn) | 0.9M | 0.34 |

Rough guide, **not measured on a GPU yet** (start with `--quick`, which runs every stage on a small
subset; on an out-of-memory error lower `--teacher-bs`):

- **16 GB cards without bf16 (V100):** `yolox_s`, `--teacher-bs 32`. The teacher stage trains in fp32
  there (bf16 autocast needs compute capability 8.0+).
- **Ampere and newer, 24 GB+ or unified memory (e.g. DGX Spark / RTX Spark class):** `yolox_m` at
  `--bs 64`; `yolox_l` with `--teacher-bs 32`.
- During the student stages the teacher only runs forward without gradients (small memory); expect
  each student epoch to take roughly 1.2-1.5x as long.

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
