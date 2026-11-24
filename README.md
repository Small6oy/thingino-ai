# thingino-ai

On-camera AI detection for [Thingino](https://thingino.com/) cameras: **person, pet, package and vehicle**
(sound later), detected locally and turned into HomeKit events, on-screen boxes and MQTT, with a web
UI to switch each one on or off. No cloud. Open architectures, trained on open data.

This repo is the home for the AI work: the training pipeline, the tools, the design and the camera
integration. The integration code is developed on the upstream Thingino/raptor forks (so it can be
contributed back) and **mirrored** here as patches; see [camera/BRANCHES.md](camera/BRANCHES.md).

## Where it stands

One person / pet / package detector, trained once and running on both camera generations:

| | T31 (e.g. Wyze Cam v3) | T41 (e.g. Wyze Cam v4) |
|---|---|---|
| Runs on | the CPU, Ingenic JZDL | the NNA, Ingenic venus (int8) |
| Time per frame | ~104 ms | ~37 ms |
| CPU at 1 fps | 8.4 % (same as Ingenic's stock persondet) | not measured live yet |
| Model file | 0.9 MB | 1.0 MB (~3.9 MB of NNA memory) |
| Exactness on the camera | bit-identical to the PyTorch model | bit-identical to Ingenic's PC int8 simulator |
| AP50 person / pet / package | 0.460 / 0.449 / 0.266 | 0.442 / 0.315 / 0.153 |

AP50 on 600 validation images. The mean AP50 of the T31 model over the full 3,000-image validation
set is 0.341. Ingenic's persondet, the starting point, scores 0.250 on person and has no other classes.

On a live camera, **person works well**. **Pet and package are not good enough yet**: their scores
are low, and a real parcel on a car was not detected at all. The cause is the training data (below).
The models aren't wired into raptor yet: that is the next step ([docs/tasks.md](docs/tasks.md)).

## The network

The detector keeps the topology of Ingenic's public `persondet` model, so it runs on the JZDL runtime
that already ships for the T31, and starts from its trained weights:

- **Input:** 416x320 RGB. The camera's 640x360 sub stream is stretched to that size (no letterbox).
- **Backbone:** a 3x3 stride-2 conv, then depthwise 3x3 + pointwise 1x1 blocks with 2x2 max pools,
  widths 16 → 32 → 64 → 128 → 256 → 512.
- **Two YOLO-style heads:** stride 32 (10x13 grid) and stride 16 (20x26 grid, fed by a 2x upsample
  of the deep features concatenated with the stride-16 backbone features). 3 anchors per cell, each
  predicting box, objectness and 3 class scores (24 channels per head). Anchors: stride 32
  81x82 · 135x169 · 344x319; stride 16 10x14 · 23x27 · 37x58.
- **Decode:** YOLOv5 formula (sigmoid, `(2s - 0.5 + cell) * stride`, `(2s)^2 * anchor`),
  score = objectness × class, per-class NMS at IoU 0.45.

**Quantisation-aware training for the T31.** JZDL runs this graph in integers: 8-bit input,
4–5-bit activations, int16 accumulators, a per-channel requantise. The training graph
([train/qat/net.py](train/qat/net.py)) reproduces that arithmetic exactly, so the PyTorch model,
the exported `.bin`, our NumPy reference runtime and the camera give the same numbers, bit for bit.

**One model, two exports.** The same trained weights become a JZDL `.bin` for the T31, and a float
ONNX that Ingenic's magik-transform-tools converts to an int8 NNA model for the T41. The ONNX uses
only ops the converter accepts (the activation clamp is built from Relu + 1x1 depthwise convs, the
upsample is a ConvTranspose). Details: [docs/one-model-two-exports.md](docs/one-model-two-exports.md).

## Training: `train/build_model.py`

One command downloads everything, trains, and exports both models:

```
cd train
python build_model.py --quick     # ~5-10 min smoke test
python build_model.py             # the full build
```

It checks the machine, downloads the persondet starting weights (md5-checked) and the dataset,
self-tests the integer arithmetic, trains 30 epochs with a live status line, then exports and
verifies both models. It is resumable at every step. Full guide, install notes per GPU and
troubleshooting: [train/README.md](train/README.md).

It runs on NVIDIA (CUDA), Intel Arc (XPU), Apple Silicon (MPS) and the CPU. Full builds by the
community so far, all with the same settings and all bit-exact:

| Machine | Training speed | Total build | Best mean AP50 |
|---|---|---|---|
| Intel Arc Pro B70 | ~307 img/s | 1 h 05 m | 0.341 |
| NVIDIA Tesla V100 16 GB | ~267 img/s | 1 h 26 m | 0.321 |
| NVIDIA DGX Spark (GB10) | ~134 img/s | 2 h 26 m | 0.329 |
| Apple M2 (batch 16) | ~28 img/s | ~10 h | – |

The AP differences between machines are run-to-run noise (each run moves about ±0.02 from epoch to
epoch); the GPU sets the speed, not the accuracy.

The T41 conversion needs Docker on an x86-64 machine (Ingenic's converter is an x86-64 program).
Elsewhere the build stops at `model_t41.onnx` plus `checkpoint.pt`, which is all that's needed to
convert later.

## Data

Dataset v2: ~76,000 images, all open data, listed in [train/manifest/](train/manifest/) (image id +
boxes, classes in `classes.json`). COCO and Open Images images are downloaded from their original
hosts at build time; the Roboflow Universe images come as one pinned bundle on this repo's
[releases](https://github.com/Small6oy/thingino-ai/releases/tag/data-roboflow-v1), so nobody needs a
Roboflow account.

| Class | Sources |
|---|---|
| person | COCO 2017, Open Images, CatoCam (home CCTV) |
| pet (cat + dog) | COCO 2017, Open Images (many outdoor), CatoCam (home CCTV, top-down, IR night) |
| package | Open Images "Box", Roboflow Universe parcel sets (doorsteps, porches, deliveries, conveyors) |
| vehicle (car, truck, bus, van, motorcycle) | COCO 2017, Open Images, a security-camera packages + vehicles set |

69,854 training and 6,223 validation images. Every image is labelled for all four classes wherever
its source annotates them; for Roboflow sets that label one class only, the missing classes were
filled in by a large COCO detector (torchvision Faster R-CNN, score >= 0.7; see
[train/data/](train/data/)). Licences: COCO and Open Images annotations CC BY 4.0; the Roboflow sets
public domain, CC BY 4.0 or MIT (list with authors in the release's `ATTRIBUTION.md`). Images keep
their own licences.

Dataset v1 (person / pet / package, 37,600 images) is in git history; its models are the current
`v1` in the model library.

**Still weak:** packages and pets seen small, from above and at night are under-represented, and none
of the data comes from our own cameras yet. Labelled frames from real Thingino cameras are the next
step after v2.

## Models

Trained models are **not published**: they are fine-tuned from Ingenic's persondet weights. They
live in a private library; [models/](models/) describes its layout and lists what exists in
[models/catalog.json](models/catalog.json). Anyone can rebuild them with `train/build_model.py`.

## Pipeline (planned)

```
frame / audio → backend (JZDL · NNA · audio CNN) → NMS + tracking gate
   → event {class, box, score} → rmd (per-class region, mask, cooldown)
      → split, per class and independent:  notify (MQTT)  ·  record (HomeKit Secure Video)
         → rhkd HomeKit sensors  +  on-screen boxes
```

Two rules borrowed from a shipping commercial camera on the same T31 silicon (the Aqara G100): a
per-object **tracking gate** before any report, and **detect / notify / record** as three independent
per-class decisions.

## Layout

```
train/        build_model.py: dataset, QAT training, T31 + T41 export (qat/, t41/, manifest/)
tools/        jzdl (T31 .bin parser/writer + bit-exact NumPy reference); more tools are moving in
models/       model library layout + catalog (model files live in a private companion repo)
camera/       thingino integration, MIRRORED from the forks: patches/ + BRANCHES.md
docs/         design: ai.md, tasks.md, benchmark.md, one-model-two-exports.md, the G100 detection reference
data/         benchmark / calibration frame sets
```

## Public-safe

This repo is public. It contains **no IP addresses, host names, local paths or household details**,
**no Ingenic vendor binaries** (runtime libraries or model templates; tools take them by path) and
**no trained models**. `scripts/check-public-safe.sh` rejects all three; run it before every PR.
Changes go through pull requests.

## Status and roadmap

Done: the T31 model format and converter (bit-exact on the real camera), quantisation-aware
training, the one-command build, the T41 export (bit-exact on the real NNA), and live tests on both
cameras. Next: better pet / package data and per-class thresholds, then running the detector live
inside raptor. Roadmap and gates: [docs/tasks.md](docs/tasks.md).

## License

TBD. Intended open; models trained on open data only.
