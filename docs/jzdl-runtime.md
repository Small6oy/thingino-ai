# ojzdl: our own JZDL runtime for the T31

`ojzdl` is an open replacement for Ingenic's `libjzdl.m.so`. It runs JZDL v4 and v5 models (Ingenic's `persondet` and `yolov5` samples, and our person/pet/package detector) on the T31 CPU. The code is in [tools/ojzdl/](../tools/ojzdl/README.md); the model format and arithmetic are in [jzdl-format.md](jzdl-format.md).

**Why:** libjzdl already ships in Thingino and is small (609 KB), so this isn't about size. What we gain:
- **Openness:** the converter, the model and the runtime are all ours.
- **New operations:** we can add operations libjzdl lacks (our models are limited to persondet's building blocks today).
- **A fully open T31 detection path** for upstream Thingino.

The risk was speed: every millisecond of inference comes out of video and audio on a single-core T31X.

## Where it stands (2026-11-24)

| Step | State |
|---|---|
| 1. Plain C runtime, bit-exact | **Done.** Bit-identical to libjzdl on 2 models × 6 frames under QEMU, and on the camera. |
| 2. Benchmark against libjzdl | **Done.** Plain C was 2.84 s per frame on the camera, about 30× slower. |
| 3. MXU2 vector kernels | **Done. Same speed as libjzdl, slightly faster**, still bit-identical. |
| 4. yolov5 (v4 models) and a raptor-hal drop-in | **Done.** yolov5 bit-identical to libjzdl. `hal_ivs_ojzdl.c` returns the same detections as raptor's libjzdl backend, without libjzdl or libstdc++, at the same speed end to end. See "Step 4". |

On a Wyze Cam v3 (T31X) running the full Thingino stream stack, persondet, forward only, with libjzdl and ojzdl alternating in sets of 20 runs:

| Runtime | Fastest | Average |
|---|---|---|
| ojzdl | 84–90 ms | 99–100 ms |
| libjzdl 5.4.0 | 89–96 ms | 104–106 ms |

Our person/pet/package model (24-channel heads) runs at the same speed as persondet: 94 ms fastest, against libjzdl's 94 ms.

## How exactness is checked

`tools/ojzdl/gate.sh` runs every model × frame through four runtimes and requires the output to be identical byte for byte:
- libjzdl's output (`jzdlrun`, under QEMU with `-cpu XBurstR2`);
- ojzdl built for MIPS, under the same QEMU;
- ojzdl built natively (the plain C path);
- the NumPy reference `jzdl_ref.py`.

The frames are two real ones (a COCO image and a camera frame) plus four synthetic ones that push the clamps: uniform noise, all black, all white and a 2-pixel checkerboard. The result is **36/36 bit-identical**: 11,707 output floats for persondet and 15,607 for our model. On the camera, ojzdl and libjzdl produce the same output md5 as the QEMU run on both real frames, for both models.

The gate was re-run after every optimisation, and every camera build was checked the same way.

## Step 1: the plain C runtime

- C99 with no dependencies, built with `-Wall -Wextra -Werror -Wshadow -Wstrict-prototypes -Wformat=2`.
- **Loading** validates every field the runtime relies on: counts, ranges, weight sizes, shapes, the chain of activation widths, and graph order.
- **Layer set:** it runs persondet's layer types: 16, 46, 73, 74, 53, 33, 75, 69 (mode 4), 71, 51 and 76. Any other type or mode is refused at load time, never guessed.
- **Arithmetic** follows [jzdl-format.md](jzdl-format.md) "Arithmetic":
  - activations are signed-offset int8, and padding uses the activation zero;
  - first conv: Q31 with int64 math;
  - 1×1 and depthwise: `wrap16(acc + bias)`, an int16 multiply with `>> 15` (floor), a rounding shift and the ReLU clamp;
  - heads: `float32(acc + bias) * scale`, built with `-ffp-contract=off` so no FMA changes the rounding.

## Step 3: MXU2 kernels

The T31's SIMD unit is MXU2: 32 × 128-bit registers on COP2. The toolchain has no MXU2 support, so each instruction is emitted as a `.word`.
- **Encodings** come from the field layouts in QEMU's `target/mips/tcg/mxu2.decode` (the Ingenic-focused gtxaspec fork).
- **Semantics** come from that fork's `mxu2_helper.c`, which follows the "MXU2 Programming Manual rev 2.16" and silicon where they differ.
- **`mxu2test`** checks every instruction form the kernels use against a C model, on 2,000 random vectors each, plus unaligned load/store offsets. It passes under QEMU and on the camera.

libjzdl's code was only scanned to see which instructions it uses (`tools/ojzdl/mxu2dis.py`). The kernels are written from the semantics, not copied.

**The arithmetic maps one to one:**
- `DADDS.h`: signed byte pairs summed into wrapping 16-bit lanes. This is exactly libjzdl's weight block (16 outputs × 2 inputs = 32 bytes) and its int16 accumulator.
- `MULQ.h` is `sat16((a·m) >> 15)`, the floor multiply. It differs only when a = m = −32768.
- `SRAR` is the rounding shift.

**Fallback to C.** Each kernel's limits are checked per layer at load time, and a layer outside them uses the C path, so the output never depends on which path ran. The limits:
- **1×1:** channels multiple of 16, shifts 0..15.
- **Depthwise:** stride 1, pad 1, the pad slot's weight 0.
- **First conv:** 3×3 × 3 channels, padding-slot weights 0, and a weight bound that keeps the int16 accumulator from wrapping.
- **Heads:** 16-bit pair sums.

**Kernels:**

| Layer | Scheme |
|---|---|
| 1×1 (53, 74) | 4 pixels × 16 outputs per block. Each weight block is loaded once per input pair; `REPI.h` broadcasts each pixel's input pair; 8 `DADDS.h` per pair. Requant: `ADD`, `MULQ.h`, `SRAR.h`, clamp, `SHUFV` narrow, store. |
| 1×1 + pool (74) | Fused: convolve two input rows into a small scratch buffer, then `MAXS.b` pools them into one output row. |
| Depthwise (73) | Padded input copy; the 10 weight vectors stay in registers per 16 channels. Per pixel: 9 tap loads, `SHUFV` interleaves the tap pairs, then `DOTPS`/`DADDS` and the same requant. |
| First conv (46) | Each kernel row is 9 contiguous HWC bytes, so it is the 1×1 scheme with rows as inputs. The Q31 requant is vectorised too: `DOTPS.w` widens even and odd outputs to words, then `ADD`, `MULQ.w`, `SRAR.w`, clamp, and `SHUFV` + `ORV` narrow. |
| Heads (51) | `DOTPS.h`, then `DADDS.w` against a ones vector for an exact int32 sum, then the float scale. |
| Pool (75), input | `MAXS.b`; the input's `rgb − 128` is `rgb XOR 0x80` (`XORIB`). |

**Memory:** one arena holds every activation, with offsets planned at load from each blob's live range, plus one scratch buffer. Both are touched once at load time, so a frame does no allocation and takes no page faults. persondet needs a 1.46 MB arena and 0.54 MB of scratch.

**Progress on the camera** (persondet forward, fastest of 5–20 runs):

| Version | Forward |
|---|---|
| Plain C | 2,837 ms |
| + MXU2 1×1 | 1,228 ms |
| + depthwise, first-conv accumulation | 212 ms |
| + arena, fused pool, heads | 141 ms |
| + first-conv vector requant | 89 ms |
| + vector input conversion | **84–90 ms** |

**Per layer now:**

| Kind | Time |
|---|---|
| 1×1 | 33.4 ms |
| 1×1 + pool | 11.7 ms |
| depthwise | 12.5 ms |
| first conv | 5.9 ms |
| heads | 2.7 ms |
| the rest | 0.7 ms |

QEMU numbers are not a guide to the camera: QEMU emulates MXU2 with helper calls and runs scalar code cheaply. Profile on the camera.

`tools/jzdl/jzdlbench.cpp` times libjzdl the same way `ojzdlrun -n` times ojzdl: input + run, N times, after one load.

## Step 4: yolov5 (v4 models) and a raptor-hal backend (2026-11-24)

**4a. Version-4 models.** Upstream Thingino's detection option (`IVS_DETECT`) ships Ingenic's `magik_model_yolov5.bin`, a version-4 model. ojzdl now runs it, and it is **bit-identical to libjzdl**: 54/54 in the gate (3 models × 6 frames × 3 runners) and both real frames on the v3. The rules are in [jzdl-format.md](jzdl-format.md) "Version 4".
- v4 weights are stored plain. ojzdl repacks them at load into the v5 block layout, so one set of MXU2 kernels serves both versions.
- The depthwise kernel gained stride 2 and asymmetric pads. Type 78 and nearest-neighbour upsample run in C.
- On the v3, yolov5's forward takes 184–187 ms with ojzdl and 175–183 ms with libjzdl (fastest of 10): within 3–5 %.

**4b. raptor-hal backend.** `src/hal_ivs_ojzdl.c` implements `hal_jzdl_create` / `_detect` / `_destroy` (the API rvd calls) on ojzdl, built with `OJZDL=1` instead of `JZDL_INCLUDE`. It is a drop-in: rvd doesn't change, and the hal no longer needs C++, libjzdl.m.so (609 KB) or libstdc++ (2.2 MB).
- **Preprocessing** is exact. `ojzdl_resize` reproduces `jzdl::resize` (see "Preprocessing" in jzdl-format.md), and NV12 is converted to RGB one row at a time with the C++ backend's formula, so there is no full-frame RGB buffer.
- **Decode and NMS** are the same code as `hal_ivs_jzdl.cpp`, plus one fix: a model whose output is smaller than the 3-head YOLOv5 decode reads is rejected with an error. The C++ backend reads out of bounds instead; with persondet it segfaults.
- **Checked:** `tools/ojzdl/haltest.c` drives both backends on real 640×360 NV12 frames. Detections are identical in every box and confidence bit (52 boxes under QEMU, 40 on the v3).
- **Speed on the v3, end to end** (yolov5, `hal_jzdl_detect`, fastest of 12):

| Stage | C++ + libjzdl | ojzdl backend |
|---|---|---|
| NV12 → RGB + resize | 19.6 ms | **16.5 ms** |
| inference | 179–181 ms | 192–194 ms |
| decode (same code) | 22–24 ms | 22–23 ms |
| total | 243–246 ms | 247–248 ms |

- The ojzdl objects are built at `-O2`. At raptor-hal's default `-Os`, inference was 205 ms.

**4c. Live test in rvd on the v3 (2026-11-24).**
- **Build:** rvd from the soak's raptor source (db94990, N-stream) with `IVS_DETECT=1 JZDL_LIBS=`, using raptor branch `ivs-jzdl-libs`, which makes the JZDL link libraries a variable. It is linked against the `OJZDL=1` hal and built with xb's SDK and make line in a private tree. rvd is 232,816 B, and NEEDED has **no libjzdl.m.so and no libstdc++**.
- **Config:** `[motion] algorithm = yolo`, `model = /tmp/…/magik_model_yolov5.bin`, defaults otherwise (conf 0.40).
- **Result:**
  - rvd logged `JZDL: model loaded (320x416x3, ojzdl)`, `ready (640x360 -> 320x416, 4 classes)` and `JZDL inference thread started`.
  - `raptorctl rvd ivs-status` showed active/persondet. There were 0 detections at 0.40, as expected given the sample model's scores.
  - **Video unaffected:** main 20.0 fps, sub 15.0 fps (measured on the rings).
  - rvd used ~75 % CPU and the system 5 % idle. Upstream's JZDL thread runs on every frame, with no skip or motion gating, so it saturates the single core; libjzdl would too.
  - rvd's RSS was 12.1 MB, flat over the run (7.5 MB without detection). No errors or warnings, same pid throughout.
- **Not done on purpose:** a very low threshold to force detections. It would have fired HomeKit motion events. Box-for-box equality is already proven by `haltest` on the same frames.

**Findings about upstream's yolov5 path** (identical with libjzdl, so not ojzdl's doing):
- On COCO frames full of people, the best confidence is about 0.06, far below rvd's default threshold of 0.40. Even fed directly, the sample model scores below 0.01 on a squashed COCO frame. The likely cause is the transposed input header (the runtime sees a 320-wide, 416-tall image while the camera frame is 640×360, so the image is squashed). Worth raising upstream with the detection work.
- Any non-yolov5 model, such as persondet or our person/pet/package model, crashes the C++ backend (out-of-bounds decode). Ours refuses it cleanly. Those models need their own decode, which the AI task list already plans ("multi-class JZDL in raptor-hal").

## Next

- **Upstream:** propose the raptor-hal backend and a Buildroot option to choose it.
- **yolov5 speed:** about 13 ms behind libjzdl. Its many narrow 1×1 layers and the add (still C) are the candidates.
- **Speed:** the 1×1 layers are 45 of the ~67 ms. Try 8-pixel blocks and row tiling for L1.
- **Memory:**
  - give blobs a pad ring inside the arena, so depthwise needs no padded copy;
  - drop the C-path weight copies for layers that run on MXU2.
