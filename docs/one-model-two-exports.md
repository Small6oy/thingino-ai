# One model, two exports

The person / pet / package detector is trained **once** and exported **twice**: a JZDL model for
T31 cameras (CPU) and an NNA model for T41 cameras. Both come out of the same training run and the
same weights, so the two camera generations detect the same classes with the same network and the
same decoding.

```
                         train/build_model.py
                                  │
            COCO + Open Images ── QAT training (persondet graph, T31 integer rules)
                                  │
                       trained weights (BatchNorm folded)
                    ┌─────────────┴──────────────┐
            T31 export (exact)            T41 export (float → int8)
         qat/export.py                  qat/export_t41.py
         model_t31.bin (JZDL)           model_t41.onnx
         runs on the CPU (MXU2)                 │  Ingenic magik-transform-tools 2.0.4
         == PyTorch, bit for bit                │  post-training int8, calibration frames
                                        model_t41.bin (venus, NNA)
```

## The two exports

| | T31 (e.g. Wyze Cam v3) | T41 (e.g. Wyze Cam v4) |
|---|---|---|
| Runtime | JZDL (`libjzdl.m.so`) on the CPU | venus (`libvenus.so`) on the NNA, CPU0 |
| File | `model_t31.bin`, ~0.9 MB | `model_t41.bin`, ~0.9 MB (from `model_t41.onnx`, ~3.5 MB float) |
| Quantisation | done in training (QAT): 8-bit input, 4–5-bit activations, int16 accumulators, per-channel requant | done by Ingenic's converter after training (PTQ): int8, calibrated on sample frames |
| Exactness | the exported model equals the PyTorch model **bit for bit** (checked by `build_model.py`) | the ONNX equals its PyTorch float twin to ~1e-5; the int8 model approximates that |
| Input | 416x320 RGB (16:9 frame stretched) | same |
| Output | two heads, 3 anchors x (5 + 3) logits, stride 32 and 16 | same heads, same decode |

The decoder (sigmoid, YOLOv5-style box formula, the two anchor sets, per-class NMS) is shared. Only
the backend that runs the network differs.

## What the T41 export changes, and why

The T41 export starts from the same folded weights as the T31 export, with three changes forced by
what Ingenic's converter accepts (each was tested against magik-transform-tools 2.0.4):

| T31 graph | T41 ONNX | Why |
|---|---|---|
| activation = `clamp(round(y / s), 0, t) * s`, `t = 2^b - 1` | `clamp(y, 0, t·s)` built as `Relu` → depthwise 1x1 `Conv` (w = −1, b = t·s) → `Relu` → the same `Conv` again, i.e. `t·s − relu(t·s − relu(y))` | The NNA quantises to int8 itself, so the rounding goes. The upper clamp must stay: without it AP50 falls to ~0, because the model learned to rely on those ceilings. The converter rejects `Clip` in every form (opset 10 attributes, opset 11+ inputs), so the clamp is built from ops it accepts. Exact, at the cost of two cheap 1x1 depthwise layers per activation. |
| zero-insertion upsample (stride 16 branch) | depthwise `ConvTranspose`, 2x2 kernel `[[1, 0], [0, 0]]`, stride 2 | Exact. `Resize` + `Mul` by a 0/1 mask is also exact, but the converter fails on the `Mul` (`MAGIK_CODE_NOT_MATCHED_DATA`). |
| first conv: pad right/bottom by 1, 3x3 stride 2 | `Conv` with `pads=[0, 0, 1, 1]` | Accepted as is. |

Every node carries all its attributes explicitly: the converter does not apply ONNX defaults
(`MAGIK_CODE_NOT_FIND_REQUIRED_ATTRIBUTE`). The graph is opset 12 and is written node by node from the
weights (no `torch.onnx` exporter), so it stays the same whatever PyTorch version trained it.

The cost of the T41 path is measured, not assumed. `build_model.py` reports, on the same
validation frames:

- **T31 exact**: the trained model as the T31 runs it.
- **T41 float**: the ONNX's PyTorch twin (what the converter starts from).
- **T41 int8**: the converted model run by Ingenic's PC executor (`magikExecutor`), an int8
  simulation of the NNA. It also prints the cosine similarity of the int8 outputs to the float ones;
  Ingenic's guide treats > 0.98 as good.

## Calibration

The converter picks each tensor's int8 range from calibration frames. `build_model.py` writes 100
validation images (16:9 crop, 416x320) to `t41/calib/`. Frames from the camera that will run the
model are better, especially night IR, which COCO hardly covers:

```
python build_model.py --calib path/to/camera/frames
```

Any `.jpg`/`.png` works; each is cropped to 16:9 and resized like the camera's sub stream.

## Running the converter

The converter is an x86-64 Linux program and is not part of this repo. `train/t41/Dockerfile`
downloads it, and Ingenic's PC executor for the int8 check, from the public
[magik-toolkit](https://github.com/wispytrace/magik-toolkit) mirror, pinned to a commit and checked
by md5, and builds `mgkrun` (`train/t41/mgkrun.cc`), our batch runner on that executor. `build_model.py` builds that image and converts automatically
when Docker is available on an x86-64 machine. Elsewhere (macOS, ARM boards such as a DGX Spark) it
stops at `model_t41.onnx`, which is all that's needed to convert later:

```
cd out/full/t41
docker build -t thingino-ai-magik:2.0.4 ../../../t41
docker run --rm -v "$PWD":/work thingino-ai-magik:2.0.4 sh -c \
  'magik-transform-tools --inputpath model_t41.onnx --outputpath model_t41.mk.h \
     --config magik_t41.cfg --save_quantize_model true > convert.log 2>&1; true'
```

The tool exits with status 1 even when it works: check for `model_t41.bin`.

## On the camera (T41)

- venus's NN instructions only run on **CPU0**: the inference thread pins itself there.
- The NNA needs a reserved `nmem` region and the L2/ORAM split set before Linux starts. nmem use
  follows input resolution, so this 416x320 model needs much less than a 640x384 YOLOv5n.

## Later: a bigger T41 model

The NNA has headroom the T31 doesn't. Once both exports run on cameras, a larger network trained
on the same dataset (`train/manifest/`) for the T41 alone is the next accuracy step. Its export is
the same ONNX → converter path.

## Measured results

First measurement, 2026-11-23, on a mid-training checkpoint (epoch 9 of 30, so absolute numbers
are low; the gaps between rows are the point). AP50 on the validation set:

| Model | Frames | Person | Pet | Package | Mean |
|---|---|---|---|---|---|
| T31 exact (as trained) | 3000 | 0.367 | 0.205 | 0.161 | 0.244 |
| T41 float ONNX | 3000 | 0.366 | 0.178 | 0.122 | 0.222 |
| T41 float, upper clamp dropped | 3000 | 0.000 | 0.015 | 0.000 | 0.005 |
| T41 float ONNX | 1200 | 0.350 | 0.155 | 0.220 | 0.242 |
| T41 int8 (NNA simulation) | 1200 | 0.339 | 0.160 | 0.184 | 0.228 |

- The int8 conversion itself costs little: outputs match the float model with cosine similarity
  0.991 (stride 32) and 0.996 (stride 16), and AP50 moves by about 0.01.
- The bigger gap is T31 → T41 float: the model was trained with its activations rounded to 4–5
  bits, and the float graph drops that rounding. Person is unaffected; pet and package lose a few
  points. If that gap holds on the finished model, a short float fine-tune of the same weights
  before the T41 export is the fix to try first.
- Both converted models are about the same size: `model_t31.bin` 0.90 MB, `model_t41.bin` 1.0 MB.
