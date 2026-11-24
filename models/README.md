# models

The model library: trained, converted models ready to put on a camera. This folder holds the
**layout and the catalog** ([catalog.json](catalog.json)); the model files themselves live in a
private companion repo, because the current models are fine-tuned from Ingenic's persondet weights
and are not published. Anyone can rebuild them with [train/build_model.py](../train/README.md).

## Catalog

| Model | Chips | Runtime | Classes | Status | AP50 person / pet / package / vehicle (mean) |
|---|---|---|---|---|---|
| `t31-jzdl/detect-person-pet-package-416x320/v1` | T31 | JZDL (CPU) | person, pet, package | candidate | 0.439 / 0.371 / 0.212 (0.341) |
| `t41-venus/detect-person-pet-package-416x320/v1` | T41 | venus (NNA) | person, pet, package | candidate | export of the same weights; int8 0.442 / 0.315 / 0.153 |
| `t31-jzdl/detect-person-pet-package-416x320/v2` | T31 | JZDL (CPU) | person, pet, package, vehicle | candidate | 0.440 / 0.556 / 0.592 / 0.266 (0.463) |
| `t41-venus/detect-person-pet-package-416x320/v2` | T41 | venus (NNA) | person, pet, package, vehicle | candidate | export of the v2 weights; int8 0.454 / 0.637 / 0.554 / 0.282 |

The T31 v1 model is bit-identical to the integer reference under QEMU and on a Wyze Cam v3, and runs as
fast as persondet. AP50: 3,000 validation frames, COCO-style 101-point AP.
v2 adds a vehicle class and the data-v2 sets; its T31 AP50 is on the full data-v2 validation set (6,223
frames), so it does not compare directly with v1. It runs live in raptor rvd on a Wyze Cam v3 at 1 Hz
for about 7 % CPU. T41 int8 figures: Ingenic's PC executor on 400 (v2) or 600 (v1) validation frames.

## Layout

```
catalog.json                                  index of every model
<soc>-<runtime>/                              one folder per chip family + inference runtime
  <task>-<classes>-<input>/                   one model family: same task, classes and input size
    v<N>/                                     a version; a new version = a new recipe or dataset
      model.bin                               the file the camera loads
      model.json                              the card: classes, input, outputs, decode, training,
                                              metrics, verification, measured performance, md5
      train-report.json, train-log.txt        what the training run wrote
      runs/<machine>/                         other runs of the same recipe, kept for comparison
```

| Folder | Chips | Runtime |
|---|---|---|
| `t31-jzdl/` | T31 (e.g. Wyze Cam v3) | JZDL, `libjzdl.m.so`, on the CPU |
| `t41-venus/` | T41 (e.g. Wyze Cam v4) | venus, `libvenus.so`, on the NNA |

New chip/runtime pairs get their own folder (`t23-jzdl/`, `t40-venus/`, ...).

Naming:

- `<task>`: `detect`, `classify`, `sound`, ...
- `<classes>`: class names in output order, joined with `-` (or a short family name such as `coco80`).
- `<input>`: network input `WIDTHxHEIGHT`.
- `v<N>`: bumped when the recipe, dataset or graph changes. Retraining the same recipe on another
  machine is not a new version: the best run becomes `model.bin`, the others go under `runs/`.
- The same weights exported for two runtimes keep the same family and version in both folders.

Card `status`: `default` (what a camera should load), `candidate` (verified, not yet run live),
`experimental`, `deprecated`.

Model files (`.bin`, `.onnx`, `.mgk`, `.pt`) must not be committed here; `.gitignore` and
`scripts/check-public-safe.sh` refuse them.
