# JZDL model format (T31, libjzdl 5.4.0 "simd", MXU2)

Goal: our own converter (PyTorch → JZDL `.bin`) so the v3 (T31) can run models we train ourselves, e.g. person + pet + package. Plain ncnn on the v3 is 2–25 s per frame, so JZDL is the only CPU runtime fast enough on the T31. Policy: Ingenic's free sample models (gtxaspec/magik_models) are fine to study. Aqara's models are **not** to be decrypted or reused.

Reference files in `tools/aitest/out/v3/`:

| File | Size | MD5 |
|---|---|---|
| `libjzdl.m.so` (5.4.0 simd) | 623,680 | `17afd09ecff1f7f90d463bab6e33bfda` |
| `magik_model_persondet.bin` (version 5, 1 class) | 897,852 | `895852133638010d3f3b23fc35e0383b` |
| `magik_model_yolov5.bin` (version 4, 4 classes) | 870,268 | `519b9df88aabc31a7458289406ef4a9f` |

Tools: [tools/jzdl/](../tools/jzdl/README.md).

- `jzdl.py` parses and writes models.
- `jzdl_loadcheck.py` runs libjzdl's own loader under Unicorn on the Mac.
- `jzdl_ref.py` is the bit-exact NumPy forward pass.
- `probe.py` exposes any intermediate blob.
- `jzdlrun.cpp` runs a model on one raw frame under QEMU or on the v3.
- Our own runtime, [tools/ojzdl/](../tools/ojzdl/README.md), is bit-identical to libjzdl (2026-11-24): [jzdl-runtime.md](jzdl-runtime.md).

The offline runtime is described under "Offline runtime" below.

## Status

| Step | State |
|---|---|
| 1. Parser + writer, byte-identical | **Done 2026-11-23.** Both samples round-trip byte for byte. Every blob offset matches the real loader's freads (100/100 persondet, 353/353 yolov5). |
| 2. Integer rebuild vs runtime output | **Done 2026-11-23 against the emulated runtime.** `jzdl_ref.py` reproduces persondet bit for bit from the raw RGB frame: every blob, and all 11,707 packed output floats, on two frames (COCO + a v3 camera frame). See "Arithmetic". **Confirmed on the real v3 on 2026-11-23 10:55:** jzdlrun + persondet on both frames gave output bit-identical to `jzdl_ref.py` and to QEMU, at 0.13 s per run (load + inference). |
| 3. Edit → runtime changes as predicted | **Done offline 2026-11-23.** One hidden 1×1 weight, one depthwise multiplier and one head bias were edited. The reference predicted 6,184 changed output values, and the emulated runtime matched bit for bit. The on-camera repeat is still to do. |
| 4. QAT person + pet + package | **Pipeline proven 2026-11-23** (now `train/qat/`). `net.py` is the persondet graph in PyTorch with fake quantisation and 3 × (5 + N) heads. `export.py` writes the `.bin`, and `test_pipeline.py` checks it. An untrained 3-class export (902,556 B, only 4.7 KB more than persondet) loads OK. QEMU output equals `jzdl_ref` bitwise on 4 frames. The training forward emulates the hardware requant exactly: PyTorch == `jzdl_ref` == runtime, bitwise. Data: COCO person/cat/dog + Open Images Box (`data_prep.py` on the build host). `train.py` (M2/MPS) starts from persondet; the first full run takes about 12 h. Deploy cost on the v3 (no JZDL in the current image; overlay has 8.2 MB free): model 0.9 MB + libjzdl 0.6 MB + libstdc++ 2.2 MB. Bake them into the squashfs rather than the overlay. |

## Design target: multi-class heads

The model we train is **multi-class in one network**, like the Aqara G100 appears to be (four per-class thresholds printed at its startup: person 0.20, pet 0.70, package 0.50, car 0.60; not yet proven multi-class, but we build ours this way regardless). Same backbone and two heads as persondet; only the head width changes.

- Head output channels = `num_anchors × (5 + num_classes)`.
  - persondet: 3 × (5 + 1) = 18 (what we have).
  - **our target: 3 × (5 + 3) = 24**, classes = person, pet, package. (4 classes → 27 if we add car later.)
- Use clean `3 anchors × (5 + N)`. Do **not** copy Aqara's suspected 18-channel squeeze (2 anchors × 4 classes); that only exists to keep their file the sample's size, a constraint we don't have.
- **Converter impact: minimal.** The `type-51` head writer already emits an arbitrary-width head with per-channel int32 bias + float32 scale (it round-trips persondet's 18-channel heads byte for byte). More classes = larger `num_output` → more weight bytes (`C_in × C_out`) and more bias/scale entries, all sizes the writer already handles. **No new layer types, no structural change to `jzdl.py`.**
- **Only real code touch is post-processing, not the writer:** the decode must read `5 + N` values per anchor (aitest's `headed_decode` assumes 6 = 1 class). Generalise it to N class scores.
- Multi-class rides on the same pieces single-class needs: the requant arithmetic (step 2, solved 2026-11-23 for the persondet layer set) and the QAT pipeline (step 4, open). It adds a config parameter, not new machinery.
- Data: person + pet (COCO person; cat/dog) are easy; package needs a parcel set (Open Images has suitable classes).

## How the format was confirmed

The Net/Layer code in libjzdl is ncnn-derived: a layer creator table, `load_param` and `load_model` per layer, and the strings "layer load_model %d failed" and "load_param_bin". The internals are stripped, so the methods were found from the data instead:

- **Creator table:** `0x90b24`, 86 entries indexed by layer type.
- **Path to the methods:** creator → constructor → vtable, with `load_param` at vtable+8 and `load_model` at vtable+0xc.
- **Net vtable:** `Net::load_model(path, bool)` is at `0xaaac`, and the param parser at `0xa054`.

`jzdl_loadcheck.py` runs `net_create()` and `load_model()` under Unicorn with libc stubbed and logs every `fread`. Every branch in the `load_param` readers tests the **version** only, never a field value, so the number of param words is fixed per (type, version).

## File layout (confirmed)

All int32, little-endian:

```
param_bytes                  # bytes from offset 24 to the end of the param section
input_blob                   # 0   (Net+0x38, get_model_input_index)
output_blob                  # 29  (Net+0x3c, get_model_output_index)
in_h, in_w, in_c             # 320 416 3 (persondet); yolov5 is 416 320 3
layer_count | version << 24  # 0x0500001c = 28 layers, version 5; yolov5 0x04000070 = 112, v4
blob_count                   # 30
per layer:  type, n_bottom, n_top, bottom[n_bottom], top[n_top], params[N(type, version)]
then, per layer in order: that layer's data blobs (no padding, no alignment)
```

The in-memory loader parses from `buf + 24` and checks that it consumed exactly `param_bytes`. The file loader reads the same fields with `fread`.

## Layer types

Type names come from layer behaviour and graph position; the param counts are confirmed.

| Type | Role | Params v5 / v4 | Data blobs |
|---|---|---|---|
| 16 | Input | 3 / 3 (`3 67 67`, meaning unknown) | none |
| 46 | Full k×k conv: the first layer on the uint8 image, or **anywhere in the graph** (see "Type 46 as a general conv") | 19 / 18 | weights, then 3 × int32[C] |
| 73 | Depthwise 3×3 conv | 18 / 17 | weights, then 4 × int16[C] |
| 74 | 1×1 conv **+ 2×2/2 max-pool** (conv → requant → pool) | 18 / – | weights, then 4 × int16[C] |
| 53 | 1×1 conv | 18 / 17 | weights, then 4 × int16[C] |
| 51 | 1×1 head conv, float output (dtype 21); any C_out | 17 / 16 | weights `[out][in]`, int32 bias[C], float32 scale[C] |
| 33 | Split (1 → 2 blobs) | 1 / 1 (dtype) | none |
| 71 | Concat | 1 / 1 (dtype) | none |
| 69 | Upsample. Mode 4 with f4 = 0 (persondet) = **zero insertion**: `out[2i, 2j] = x`, else the activation zero. Modes 4 and 5 with f4 = 1 (yolov5) = **nearest neighbour** (verified 2026-11-24) | 5 / 5 (`20 4 2 2 0`, `20 4 2 2 1`, `20 5 2 2 1`) | none |
| 75 | Pool, 2×2 stride 2 | 12 / ? | none |
| 76 | Output pack (heads → one Mat) | 0 / 0 | none |
| 78 | Eltwise add (yolov5 residual): **mean** of the two unsigned inputs, rounded half to even; param 0 = bits_out (verified 2026-11-24 for 5+5 → 5 bits) | – / 2 (`5 20`) | none |

Types in the creator table but not in either sample: 3, 5, 6, 10, 14, 21, 23, 26, 30, 32, 39, 42–45, 47–50, 52, 56–68, 70, 72, 77, 79–85.

**Conv params, version 5.** The order below is the fread order, followed by the layer-struct offset each value lands in. These are types 73 and 46.

| # | Field | Example (73 / 46) |
|---|---|---|
| 0 | num_output C (+0x58) | 16 / 16 |
| 1–2 | kernel h, w | 3 3 |
| 3–4 | dilation h, w: **ignored by libjzdl** for 46 (verified) | 1 1 |
| 5–6 | stride h, w | 1 1 / 2 2 |
| 7–11 | pad: mode, then 4 pads (top, bottom, left, right). **46 ignores them** and pads TF "SAME" (verified) | `0 1 1 1 1`; stride 2 = `0 0 1 0 1`; 46 = `-233 0 1 0 1` |
| 12 | weight_bytes | 160 = 16 × 10 |
| 13 | ? | 0 |
| 14–15 | **bits_in, bits_out** (activation widths, confirmed). 46 has three: `4 8 5` = ? (no effect found), **bits_in** (8 on the image; verified mid-graph), bits_out | 5 5, 4 4, 5 4, 4 5 |
| 16 | dtype | 20 = int8 feature map; 51 heads use 21 |
| 17 | v5-only extra word | 1 on convs, 0 on heads |

**1×1 layers (53/74/51)** read `C, f1, kh, kw, f4, f5` and then a 5-word array before `weight_bytes`. In the samples these are always `C 1 1 1 1 1 0 0 0 0 0 0`. Head 51 has one word fewer after weight_bytes (`4 32` instead of `0 a b`).

**Weight sizes.**

| Layer | Version 5 | Version 4 |
|---|---|---|
| 1×1 conv | C_in × C_out bytes | same |
| Depthwise 3×3 | C × 10 (9 taps padded to 10) | C × 9 |
| First conv | C_out × 3 × 10 | C_out × 3 × 9 |

Weight values seen:

| Layer | Range | Implied width |
|---|---|---|
| 1×1 | −16..15 | 5-bit |
| Depthwise | −128..127 | 8-bit |
| First conv | −8..7 | 4-bit |
| Heads | int8 (not yet ranged) | ? |

## Arithmetic (confirmed 2026-11-23, bit-exact vs runtime)

These were found with designed experiments: tiny models written with `jzdl.py`, using one-hot weights and chosen bias, mult and shift, then run in the emulated runtime. The results were confirmed on the full persondet.

**Activations.** Every int8 feature map holds `u − 2^(bits−1)`, where `u` is the unsigned `bits`-wide output of a ReLU. For example, 5-bit activations are stored as −16..15, and −16 means 0.
- The input is `rgb − 128` (aitest/raptor run `image_sub(img, 128)`).
- Convolutions accumulate in this **signed** domain.
- Padding uses the activation zero: −2^(bits_in−1). For the image that is −128, i.e. pixel value 0.

**Requantisation** (per output channel; `rsr` = add 2^(s−1), then arithmetic shift right):

```
u   = clamp( rsr( ((acc + bias) * mult) >> Q , shift ), 0, 2^bits_out − 1 )
out = u − 2^(bits_out − 1)
```

- For the first conv (46), Q = 31, and bias, mult and shift are int32. **Its `acc` wraps to int16 before the bias is added** (see below); for the others `acc + bias` wraps.
- For every other conv (53/73/74), Q = 15, and they are int16. The fourth int16 array is unused; it is 0 in all persondet layers.
- **Their shift is taken modulo 16** (verified 2026-11-25 on a MobileNetV4 export: channels with shift 16/17 came out saturated; with `shift & 15` jzdl_ref matches libjzdl on every value). A shift of 16 acts as 0. Our exporter caps these shifts at 15 (`net.MAX_SH`) and shrinks `mult` instead; jzdl_ref and ojzdl apply `& 15` so they match libjzdl on any file.
- A type 46 inside the graph (not on the image) follows the same rules as the image conv: its padding is its input's code 0 (−2^(bits_in−1), e.g. −16 for 5 bits) and its `acc` wraps to int16 before the bias (verified 2026-11-25; ojzdl's C path had assumed the image's −128 and no wrap).
- The multiply truncates (floor). Only the shift rounds.
- Negative results become 0, which is the ReLU.

**Head (51):** `float = float32(acc + bias) × scale`, where `acc = Σ w · x` over signed activations. There is no clamp.

**Output pack (76):** each head flattened HWC, one after another, then a float trailer `h, w, c` per head and the head count.

**Weight layouts.** The MXU2 unit works on 16 output channels × 2 inputs, so the weights are stored in 32-byte blocks. In each block, `byte = o·2 + slot`.

| Type | Order |
|---|---|
| 46 first conv | Within a group of 16 outputs, tap `s = ky·10 + kx·C_in + ci`; each kernel row's 9 values are padded to 10. `byte = (s // 2)·32 + o·2 + s % 2`. |
| 73 depthwise | 160 bytes per 16 channels. Pairs `t0 t1 \| t2 t3 \| t4 t5 \| t6 t7 \| pad t8`, with **the pad before t8**. `byte = (p // 2)·32 + c·2 + p % 2`, where p = slot of the tap. |
| 53 / 74 1×1 | Blocks ordered `[output group of 16][input pair]`, each holding 16 outputs × 2 consecutive inputs. |
| 51 head | Plain `[out][in]`, no blocking, so any C_out works (18, 24, 27). |

**Accumulator: int16, wrapping (tested 2026-11-23 with full-range weights).** For 1×1 and depthwise convs the runtime computes `wrap16(acc + bias)`, i.e. modulo 2^16, before the Q15 multiply. Beyond int16, outputs flip sign. The first conv also misbehaves out of range; its exact rule wasn't decoded because it isn't needed.

Ingenic's quantiser keeps every persondet layer's **worst case** inside int16: the largest bound, Σ|w|·x_max + |bias| over all channels and layers, is 21,301. That is why its 1×1 weights use only 4–5 bits. **Export rule:** for each output channel, the exact range of `acc + bias` over unsigned inputs u ∈ [0, 2^bits_in − 1] must stay within int16: `max(|b + umax·Σw⁺|, |b + umax·Σw⁻|) ≤ 32767`, where b is the bias for unsigned inputs. Inside that bound wrap never happens, and `jzdl_ref.py` is exact. QAT must train under the same bound.

**Not covered yet:**
- type 78 with unequal input widths (yolov5 only has 5 + 5 → 5);
- the `f13` field and the Input params `3 67 67`.

## Version 4 (yolov5 sample), confirmed 2026-11-24

Pinned down by probing every intermediate blob of `magik_model_yolov5.bin` under QEMU and fitting each layer in isolation: `tools/jzdl/v4probe.py MODEL FRAME WORK --run`, then `--isolated`, which feeds each layer libjzdl's own inputs. With the rules below, `jzdl_ref.py` matches libjzdl on every blob and on all 73,720 output floats, and ojzdl does too (6 frames, under QEMU and on the v3).

- **Arithmetic: unchanged from v5.** Same requant, same int16 arrays (the fourth still 0), same Q15/Q31.
- **Params:** v4 lacks v5's trailing flag word; every other index is the same.
- **Weights are stored plain, not in MXU2 blocks:**

| Type | v4 layout |
|---|---|
| 46 first conv | `[ky][kx][ci][out]` (27 × C_out bytes, no row padding) |
| 73 depthwise | tap-major `[ky·3 + kx][c]` (9 × C bytes) |
| 53 1×1 | `[out][in]` |
| 51 head | `[out][in]`, as in v5 |

- **Depthwise stride 2** (`2 2`, pads `0 1 0 1`: top 0, bottom 1, left 0, right 1) uses the same arithmetic as stride 1.
- **wrap16 really happens in yolov5.** On noise and black frames its int16 accumulators overflow, and libjzdl wraps them as described under "Accumulator". `jzdl_ref.py` now implements the wrap; without it, 134–694 of the 73,720 outputs differed.

### Type 46 as a general conv (verified 2026-11-24)

libjzdl runs a type 46 anywhere in a graph, not only on the image. Tiny v4 models and a full TinyDet export
(ten type 46 layers, 16–64 channels, kernels 3/5/9) matched `jzdl_ref.py` under QEMU bit for bit:

- **Any C_in, any kernel size** (3×3, 5×5, 9×9 tested), stride 1 or 2. v4 weights `[ky][kx][ci][out]`.
- **P[15] = input bits.** The input codes are signed `u − 2^(bits_in−1)`; the padding is code 0 (`−2^(bits_in−1)`).
- **The pad fields are ignored: always TF "SAME".** Output `ceil(n / stride)`; total pad `(out−1)·stride + k − n`,
  with the odd one at the bottom/right. On even sizes, stride 2 = nothing top/left, one row/column bottom/right,
  so a network trained with padding 1 on both sides sees its stride-2 inputs shifted by one pixel.
- **Dilation (P[3]/P[4]) is ignored.** Store a dilated conv as its zero-stuffed kernel ((k−1)·d + 1).
- **Accumulator: int16, wrapping, before the int32 bias.** `acc = Σ w·(u − 2^(b−1))` wraps to int16, then
  `+ bias` (int32), then the Q31 requant. 0 of 245,760 outputs differ under this rule; "int32 accumulator"
  or "acc + bias wraps" leave 9,889 and 14,767 wrong. Safe bound: `2^(bits_in−1) · Σ|w| ≤ 32767` per output
  channel, so wide layers need narrow activations (a 64-channel 3×3 at 5 bits leaves Σ|w| ≤ 2048 over 576 taps).
- **Input shape:** the header holds `416 320 3` (in_h, in_w, in_c), the transpose of persondet's `320 416 3`. `get_input_shape()` returns w = 320, h = 416, so the runtime sees a 320-wide, 416-tall image.

## Preprocessing: `jzdl::resize`, confirmed 2026-11-24

raptor's JZDL backend makes the model input with libjzdl's `jzdl::resize` (INTER_LINEAR), so an exact replacement has to reproduce it. It was fitted with `tools/jzdl/resizetest.cpp` under QEMU against ncnn-style fixed-point bilinear:

- 11-bit coefficients (2048 = 1.0).
- **Horizontal:** fraction `fx = (dx + 0.5)·src/dst − 0.5` in **double**; `a0 = round((1 − fx)·2048)` and `a1 = round(fx·2048)`, each rounded half up; at the edges (clamped `sx`) the coefficient is 2048.
- **Vertical:** fraction in **float**; `b0 = round((1 − fy)·2048)`, `b1 = 2048 − b0`; at the edges the coefficient is **2047**.
- `row = (p0·a0 + p1·a1) >> 4` as int16; `out = (((b0·row0) >> 16) + ((b1·row1) >> 16) + 2) >> 2`. That is ncnn's vertical step, not OpenCV's exact `>> 22`.
- **Exact** for 640×360 → 416×320, 320×416, 416×234, 1280×720 and same-size, which covers every size raptor uses. Other ratios (700×400, 999×361, 213×120) are within 1 on 0.15–0.35 % of bytes.

## Offline runtime (QEMU, no camera)

libjzdl's SIMD code is **MXU2** (COP2, 128-bit), and the T31 reserves the MXU1 opcodes. The "MXU1" label in findings.md is wrong. libjzdl also contains about 5,000 MXU1 instructions, presumably the path for older chips such as the T20. Upstream QEMU only has MXU1, so the upstream build traps in `image_sub`. An Ingenic-focused QEMU fork based on gtxaspec/qemu implements MXU2, and its `XBurstR2` CPU model is the T31.

Setup on a Linux build host, all under `~/jzdl-qemu/`:

- **`fork/`** is a `git archive` of fork commit `6802b7bc46` (see `fork/FORK_COMMIT`).
- **`fbuild/qemu-mipsel`** is built as `mipsel-linux-user --static` inside a throwaway `debian:trixie` container, because the host lacks glib-dev.
- **`run/`** holds:
  - `jzdlrun`, built with the v3 toolchain;
  - libjzdl and the sample models;
  - a `libm.so.0 → libc.so.0` symlink, because uClibc-ng 1.0.59 merged libm into libc;
  - `batch.sh DIR`, which runs every `DIR/*.bin` against its `.rgb`.

```sh
S=~/xb/tf-v3/output/xb/host/mipsel-buildroot-linux-uclibc/sysroot
~/jzdl-qemu/fbuild/qemu-mipsel -cpu XBurstR2 -L $S -E LD_LIBRARY_PATH=~/jzdl-qemu/run \
    ~/jzdl-qemu/run/jzdlrun MODEL.bin FRAME_416x320.rgb out [blob ...]
```

One persondet frame takes about 0.4 s. `run(out, blob)` only returns float blobs. To read an int8 blob, `probe.py` writes a model that ends in an identity type-51 head on that blob.

## Next

- **v3 confirmation: done 2026-11-23.** The test left nothing behind: `/tmp/ai` was removed and the daemons were untouched. The camera's dropbear refused `scp -O` from sshpass, so the payload was streamed with `ssh 'cat > file'`. The step-3 edit can be repeated on the camera the same way if wanted, but the emulator has proven bit-exact, so it isn't needed.
- **Step 4:**
  - a PyTorch model with this layer set and QAT that simulates the rules above (signed-offset activations, per-channel Q15/Q31 requant, ReLU clamp, zero-insert upsample);
  - an exporter that packs the weight layouts;
  - multi-class heads, 3 × (5 + 3) = 24.
