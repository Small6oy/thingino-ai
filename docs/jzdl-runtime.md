# ojzdl: our own JZDL runtime for the T31

`ojzdl` is an open replacement for Ingenic's `libjzdl.m.so`. It runs JZDL v5 models (Ingenic's `persondet`, and our person/pet/package detector) on the T31 CPU. The code is in [tools/ojzdl/](../tools/ojzdl/README.md); the model format and arithmetic are in [jzdl-format.md](jzdl-format.md).

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
| 4. Drop-in for raptor-hal's JZDL backend; yolov5's extra layer types | next |

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

## Next

- **Step 4:** a drop-in for raptor-hal's JZDL backend (`hal_ivs_jzdl.cpp`), so raptor can run on ojzdl. Then yolov5's eltwise add (type 78), its upsample modes, and version-4 models, each pinned down with designed experiments against libjzdl first.
- **Speed:** the 1×1 layers are 45 of the ~67 ms. Try 8-pixel blocks and row tiling for L1.
- **Memory:**
  - give blobs a pad ring inside the arena, so depthwise needs no padded copy;
  - drop the C-path weight copies for layers that run on MXU2.
