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
| 3 | Downloads ~37,600 images (COCO + Open Images) listed in `manifest/`, resized to ≤ 640 px, ~2.5 GB | 30-60 min, depends on your connection |
| 4 | Self-test: export → `.bin` → integer reference must equal PyTorch bit for bit; GPU must match CPU | ~1 min |
| 5 | Trains 30 epochs, evaluating person / pet / package AP50 after each one and keeping the best | ~1 h (the script shows its own ETA after a minute) |
| 6 | T31 export: re-checks the best model bit for bit | ~1 min |
| 7 | T41 export: ONNX + converter config + calibration frames, float accuracy check; with Docker on x86-64 also the NNA conversion and an int8 simulation check (AP50 of T31 vs T41 float vs T41 int8 on 600 frames) | ~5-10 min (+ a one-time ~1 GB converter image) |

For comparison, the same training takes about 10 hours on an Apple M2.

Everything is resumable. If it stops (Ctrl-C, reboot, lost connection), run the same command again:
downloaded images are kept and training continues from the last finished epoch.

## Requirements

- Python 3.10 or newer
- Optional, for the T41 conversion: Docker on an x86-64 machine
- An NVIDIA GPU with a recent driver (RTX 50xx: driver 570 or newer; a DGX Spark works as it
  comes). It also runs on Apple Silicon (MPS) or the CPU, just much more slowly.
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
`--device cuda|mps|cpu`, `--jobs N` (parallel downloads), `--name NAME` (separate run),
`--work DIR` (put the 2.5 GB dataset elsewhere), `--calib DIR` (T41 calibration frames from
your camera), `--t41-convert` (try the converter on a non-x86-64 machine). Defaults on CUDA: batch 64, lr 1e-3.
`--bs 16 --lr 5e-4` reproduces the original Mac run.

## When it finishes

It prints the folder with the results, `out/full/`:

| File | What |
|---|---|
| `model_t31.bin` | the T31 model (JZDL) |
| `model_t41.onnx` | the T41 model before conversion (float) |
| `model_t41.bin` | the T41 NNA model, only if the converter ran (Docker on x86-64) |
| `report.json` | machine, settings, per-epoch accuracy, the bit-exact checks, T41 float accuracy, md5s |
| `log.txt` | the training log |
| `t41/` | converter input (`magik_t41.cfg`, `calib/`), `convert.log`, and `model_t41_quant.mgk` (the int8 model for Ingenic's PC executor) |

Send the files from the first five rows back to the person who asked you to run this (about 5 MB).
**Don't publish the models** (no commit, release or upload): they are derived from Ingenic's
persondet weights.

## Troubleshooting

| Message | Fix |
|---|---|
| `missing Python package` | activate the venv, then repeat the two `pip install` lines |
| `torch.cuda.is_available() is False` | you got a CPU-only PyTorch: `pip uninstall torch torchvision`, then install again with the `cu128` index |
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
| `manifest/{train,val}.tsv.gz` | the dataset: image id + YOLO boxes (class 0 person, 1 pet = cat/dog, 2 package = Open Images "Box") |
| `qat/net.py` | the persondet graph in PyTorch, with fake quantisation that matches libjzdl's integer arithmetic exactly |
| `qat/train.py` | data loading, YOLO loss, AP50 evaluation, the training loop |
| `qat/export.py` | T31 export: trained model → JZDL `.bin` |
| `qat/export_t41.py` | T41 export: trained model → float ONNX for Ingenic's converter |
| `t41/Dockerfile` | the converter image (downloads Ingenic's converter and PC executor from the public magik-toolkit mirror, md5-pinned) |
| `t41/mgkrun.cc` | batch runner on Ingenic's PC executor: int8 simulation of the converted model |
| `qat/init_persondet.py` | starting weights from persondet, with the heads widened from 1 to 3 classes |
| `qat/calib.py` | BatchNorm folding and activation calibration helpers |
| `qat/data_prep.py` | how the manifest's images were picked (not needed for a build) |
| `../tools/jzdl/` | the `.bin` parser/writer and the bit-exact NumPy reference runtime |

## Data and licences

Images are downloaded from their original hosts and are not redistributed here.
The boxes in `manifest/` come from [COCO 2017](https://cocodataset.org/) (annotations © COCO
Consortium, CC BY 4.0) and [Open Images V6](https://storage.googleapis.com/openimages/web/index.html)
(annotations by Google, CC BY 4.0). The images keep their own licences.
