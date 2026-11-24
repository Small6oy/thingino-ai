# train: build the person / pet / package model

One command builds the detector for Thingino cameras: **one model, two exports**
([docs/one-model-two-exports.md](../docs/one-model-two-exports.md)).

- **T31** (e.g. Wyze Cam v3): a ~0.9 MB JZDL `.bin` for the camera CPU. Training is
  quantisation-aware and bit-exact: the PyTorch model, the `.bin` and the camera give identical numbers.
- **T41** (e.g. Wyze Cam v4): the same trained model as a float ONNX, converted by Ingenic's tool
  into an NNA `.bin` (automatic when Docker runs on an x86-64 machine).

```
python build_model.py --quick     # ~5-10 min smoke test: run this first
python build_model.py             # the full build
```

## What it does

| Step | What | Time (RTX 5080, estimate) |
|---|---|---|
| 1 | Checks Python, PyTorch, the GPU and free disk | seconds |
| 2 | Downloads Ingenic's public persondet model (starting weights + graph) from [gtxaspec/magik_models](https://github.com/gtxaspec/magik_models), checks its md5 | seconds |
| 3 | Downloads ~76,000 images listed in `manifest/`: COCO + Open Images from their hosts, plus one pinned bundle of open Roboflow Universe images (714 MB) from this repo's releases; resized to ≤ 640 px, ~5.5 GB | 30-60 min, depends on your connection |
| 4 | Self-test: export → `.bin` → integer reference must equal PyTorch bit for bit; GPU must match CPU | ~1 min |
| 5 | Trains 30 epochs, evaluating AP50 per class after each one and keeping the best | ~1 h (the script shows its own ETA after a minute) |
| 6 | T31 export: re-checks the best model bit for bit | ~1 min |
| 7 | T41 export: ONNX + converter config + calibration frames, float accuracy check; with Docker on x86-64 also the NNA conversion and an int8 simulation check (AP50 of T31 vs T41 float vs T41 int8 on 600 frames) | ~5-10 min (+ a one-time ~1 GB converter image) |

For comparison, the same training takes about 10 hours on an Apple M2.

Everything is resumable. If it stops (Ctrl-C, reboot, lost connection), run the same command again:
downloaded images are kept and training continues from the last finished epoch.

## Requirements

- Python 3.10 or newer
- Optional, for the T41 conversion: Docker on an x86-64 machine
- An NVIDIA GPU with a recent driver (RTX 50xx: driver 570 or newer; a DGX Spark works as it
  comes), an Intel Arc GPU with the Intel XPU PyTorch build, or Apple Silicon (MPS). The CPU
  works, just much more slowly.
- ~4 GB free disk, an internet connection

## Install

Linux / macOS:

```
git clone https://github.com/Small6oy/thingino-ai.git
cd thingino-ai/train
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # NVIDIA (CUDA 12.8)
pip install -r requirements.txt
```

Windows (PowerShell):

```
git clone https://github.com/Small6oy/thingino-ai.git
cd thingino-ai\train
py -m venv venv
venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

RTX 50xx cards (Blackwell, `sm_120`) need a PyTorch build for CUDA 12.8 or newer, which is what the
`cu128` index gives you. On a Mac use plain `pip install torch torchvision`.

Intel Arc (XPU), using Python 3.13:

```
python3.13 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
pip install -r requirements.txt
```

This needs the Intel GPU kernel driver and the Level Zero runtime, and the user must be able to
open the render device (`/dev/dri/renderD*`). The first XPU is used by default; `--device xpu:N`
picks another when an integrated GPU is also visible.

NVIDIA DGX Spark (GB10, ARM64, CUDA 13): same Linux steps, but install PyTorch from the `cu130` index:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

If that gives you trouble, NVIDIA's PyTorch container works too: start
`nvcr.io/nvidia/pytorch` (any recent tag) with `--gpus all` and this folder mounted, then run
`pip install -r requirements.txt` and `python build_model.py` inside it.

## Run

```
python build_model.py --quick
python build_model.py
```

While it runs you get a live status line:

```
  epoch 7/30  it 310/536  640 img/s  loss 0.214  9m12s  ETA 41m05s  AP50 person 0.45 pet 0.31 package 0.22
```

The same information (stage, epoch, speed, ETA, accuracy history) is in
`work/runs/full/status.json`, updated every few seconds, so you can check it from another window.

Options: `--epochs N`, `--bs N`, `--lr X`, `--workers N` (data-loading processes),
`--device cuda|xpu|mps|cpu`, `--jobs N` (parallel downloads), `--name NAME` (separate run),
`--work DIR` (put the ~5.5 GB dataset elsewhere), `--calib DIR` (T41 calibration frames from
your camera), `--t41-convert` (try the converter on a non-x86-64 machine). Defaults on CUDA and
XPU: batch 64, lr 1e-3. `--bs 16 --lr 5e-4` reproduces the original Mac run. On an Arc Pro B70
the full build is CPU-bound on image decode, so raise `--workers` before changing the batch size.

## When it finishes

It prints the folder with the results, `out/full/`:

| File | What |
|---|---|
| `model_t31.bin` | the T31 model (JZDL) |
| `model_t41.onnx` | the T41 model before conversion (float) |
| `model_t41.bin` | the T41 NNA model, only if the converter ran (Docker on x86-64) |
| `report.json` | machine, settings, per-epoch accuracy, the bit-exact checks, T41 float accuracy, md5s |
| `log.txt` | the training log |
| `checkpoint.pt` | the trained PyTorch weights (best epoch). Both exports are made from it, so with it we can re-export later (e.g. the T41 model if your machine couldn't convert) |
| `t41/` | converter input (`magik_t41.cfg`, `calib/`), `convert.log`, and `model_t41_quant.mgk` (the int8 model for Ingenic's PC executor) |

Send `model_t31.bin`, `model_t41.onnx`, `model_t41.bin` (if present), `report.json`, `log.txt` and
`checkpoint.pt` back to the person who asked you to run this (about 9 MB).
**Don't publish the models or the checkpoint** (no commit, release or upload): they are derived from Ingenic's
persondet weights.

## Troubleshooting

| Message | Fix |
|---|---|
| `missing Python package` | activate the venv, then repeat the two `pip install` lines |
| `torch.cuda.is_available() is False` | you got a CPU-only PyTorch: `pip uninstall torch torchvision`, then install again with the `cu128` index |
| `torch.xpu.is_available() is False` | install the XPU build from the `xpu` index, and check the Intel GPU driver, Level Zero and `/dev/dri/renderD*` access |
| wrong Intel GPU | `--device xpu:N` (the first XPU is used by default) |
| `no kernels for sm_120` / `sm_121` | PyTorch is too old for the RTX 50xx / DGX Spark: install from the `cu128` (Spark: `cu130`) index |
| `downloads failed (> 3%)` | a server or the connection hiccupped: run again, it only fetches what's missing |
| `GPU forward does not match the CPU one` | send us the output. `--device cpu` works but is very slow |
| out of GPU memory | `--bs 32` |
| slow (`img/s` low, GPU mostly idle) | the CPU can't decode images fast enough: raise `--workers` |

## Files

| Path | What |
|---|---|
| `build_model.py` | the end-to-end build described above |
| `sources.json` | public download URLs and the persondet md5 |
| `manifest/{train,val}.tsv.gz` | the dataset: image id + YOLO boxes; `classes.json` lists the classes (v2: 0 person, 1 pet = cat/dog, 2 package = Open Images "Box", 3 vehicle = car/truck/bus/van/motorcycle). Built by `data/build_manifest.py` |
| `qat/net.py` | the persondet graph in PyTorch, with fake quantisation that matches libjzdl's integer arithmetic exactly |
| `qat/train.py` | data loading, YOLO loss, AP50 evaluation, the training loop |
| `qat/export.py` | T31 export: trained model → JZDL `.bin` |
| `qat/export_t41.py` | T41 export: trained model → float ONNX for Ingenic's converter |
| `t41/Dockerfile` | the converter image (downloads Ingenic's converter and PC executor from the public magik-toolkit mirror, md5-pinned) |
| `t41/mgkrun.cc` | batch runner on Ingenic's PC executor: int8 simulation of the converted model |
| `qat/init_persondet.py` | starting weights from persondet, with the heads widened from 1 class to the manifest's classes |
| `qat/calib.py` | BatchNorm folding and activation calibration helpers |
| `qat/data_prep.py` | how the manifest's images were picked (not needed for a build) |
| `../tools/jzdl/` | the `.bin` parser/writer and the bit-exact NumPy reference runtime |

## Data and licences

Images are downloaded from their original hosts and are not redistributed here.
The boxes in `manifest/` come from [COCO 2017](https://cocodataset.org/) (annotations © COCO
Consortium, CC BY 4.0) and [Open Images V6](https://storage.googleapis.com/openimages/web/index.html)
(annotations by Google, CC BY 4.0). The images keep their own licences.

## Rebuilding the dataset (maintainers)

`train/data/` holds the tools that produce `manifest/` and the Roboflow bundle. Helpers never need
them.

| Tool | What |
|---|---|
| `data/build_manifest.py SRC` | COCO + Open Images rows for every class, from the public annotation files |
| `data/fetch_roboflow.py OUT` | downloads the Roboflow Universe sets in its list. It needs a Roboflow API key (in `~/.config/roboflow/key`), checks each licence live, and keeps only public domain, CC0, CC BY or MIT. It maps class names and writes `ATTRIBUTION.md`. |
| `data/fill_labels.py OUT` | fills in the classes a Roboflow set doesn't annotate, using torchvision's COCO Faster R-CNN |

After that, append the bundle rows to `manifest/`, publish the images as a release asset, and pin its
URL and md5 in `sources.json` (`bundles`).
