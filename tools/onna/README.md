# onna: open runtime for the T41 NNA (venus-compatible)

A C runtime that executes a venus (magik-transform-tools 2.0.4) `.bin` on the T41 NNA. Its output
must be bit-identical to [../nnaop/golden.py](../nnaop/golden.py), and so to venus. Plan:
[docs/nna-runtime.md](../../docs/nna-runtime.md).
Layout and conventions follow [../ojzdl/](../ojzdl/README.md), the open T31 JZDL runtime.

| File | What |
|---|---|
| `nnacpu.h`, `nnacpu.c` | the CPU half: requant + 32-segment SiLU, head float, concat rescale, add (CPU reference of the NNA's QSUM), 5×5 maxpool, nearest ×2 |
| `test_cpu.c` | runs every op of a `golden.py` dump (`GOLDEN_DUMP=dir golden.py …`) and compares bytes |
| `nnabin.h`, `nnabin.c` | loads the venus `.bin` (conv records + W, requant record, SiLU table; add records; header concat records; type-30 concat factors) and onna's text graph, validating every field it uses; builds each conv's int8 weights `[ky][kx][oc][ic]` |
| `onnarun.c` | runs a model: `onnarun [-e cpu\|oracle\|nna] [-c] model.bin model.graph input_rgba.raw prefix` writes `prefix0.f32 …` (the heads). `-e nna` pins to CPU0 and drives the real NNA; `-c` compares every conv's sums with the CPU conv |
| `nnaeng.h`, `nnaeng.c` | the NNA conv driver: our own tiling (one row of 8 output pixels per MAC, input window into F RAM, W re-encoded per 32-oc block), the 6×6 stem as 6-row strips with two MACs, sums via CPOC + 16 NNDRD. Runs through a backend |
| `mxu3.h`, `be_mxu3.c` | the camera backend: MXU3/NN instructions as `.word` (encodings from QEMU's `mxu3.decode`), VWR word 0 for scalars, VPR 0 for data |
| `mxu3test.c` | self-test of the MXU3 data path (LAO/SAO, MFCPUW/MTCPUW); PASS under user-mode QEMU |
| `hwreplay.c` | runs a traced venus instruction stream on the real NNA and compares every read-back (silicon check of the backend) |
| `fprobe.c` | which F RAM address each output pixel reads (marker pixels + identity weights); same code on oracle and silicon |
| `oracle/` | host test oracle: `mkoracle.sh` extracts QEMU's NNA engine from your QEMU checkout at build time (GPL, never copied here); `replay.c` checks it against a venus trace; `be_oracle.c` (one level up) drives it |

The graph comes from `../nnaop/mkgraph.py model.bin model.onnx > model.graph`. It holds the network
structure and which `.bin` record each op uses; the numbers stay in the `.bin`.

```sh
GOLDEN_DUMP=../gold python3 ../nnaop/golden.py model.bin model.onnx input_rgba.raw   # test vectors
make check          # host
make ORACLE=1 onnarun && ./onnarun -e oracle -c model.bin model.graph input_rgba.raw out   # NNA driver vs oracle
make check-mips     # T41 build (xb v4) under user-mode QEMU -cpu XBurst2
```

Status, 2026-11-24:
- **Step 1:** 78/78 ops bit-identical, on the host and as MIPS under QEMU (yolov5n: 57 requant,
  3 heads, 7 adds, 6 rescales, 3 maxpools, 2 upsamples).
- **Step 2:** `onnarun` with CPU convs gives heads **byte-identical to venus** on both frames
  (COCO 322864 = the v4's own output, and the v4 night frame).
  - x86: 5.4 s.
  - MIPS under `qemu-mipsel -cpu XBurst2`: 10.3 s.
- **Step 3:** the NNA driver.
  - On the host, every conv of yolov5n through the driver against the oracle equals the CPU conv: 60/60 layers (`-e oracle -c`). Heads byte-identical to venus on both frames.
  - The oracle replays venus's traced 1.3M instructions with all 661,920 read-backs identical.
  - **In the system emulator** (QEMU 0.7.26, v4 image), `onnarun -e nna` with the real instructions gives **the same md5s as venus** on both frames.
  - It runs without `/dev/soc-nna` and without nmem.
- **Step 5 (silicon), 2026-11-24:** on a Wyze Cam v4 in its stock state (no nmem, 256K L2, `soc_nna` unused), `onnarun -e nna` gives **venus's output exactly** on both frames: 2.1 s per frame, peak RSS 12.5 MB.
  - The fix that got there: the T41's F RAM is **sliced** (2-bit slices `0x37`/4 lines apart, as QEMU models only for the T40), so each plane gets `0x37` lines and is written separately.
