# onna: open runtime for the T41 NNA (venus-compatible)

A C runtime that executes a venus (magik-transform-tools 2.0.4) `.bin` on the T41 NNA. Its output
must be bit-identical to [../nnaop/golden.py](../nnaop/golden.py), and so to venus. Plan:
[../../c-driver-plan.md](../../c-driver-plan.md). Notes: [../../open-runtime.md](../../open-runtime.md).
Layout and conventions follow [../ojzdl/](../ojzdl/README.md), the open T31 JZDL runtime.

| File | What |
|---|---|
| `nnacpu.h`, `nnacpu.c` | the CPU half: requant + 32-segment SiLU, head float, concat rescale, add (CPU reference of the NNA's QSUM), 5×5 maxpool, nearest ×2 |
| `test_cpu.c` | runs every op of a `golden.py` dump (`GOLDEN_DUMP=dir golden.py …`) and compares bytes |
| `onnaq.c` | **the runner** for every venus `.bin`: `onnaq [-e cpu\|oracle\|nna] [-c] [-T] [-n N] model.bin - frame prefix` writes `prefix.<k>.f32` (the heads). The graph comes from the `.bin`'s own op list (`-`; a `.qgraph` is still read when given). The QSUM kind (the thingino-ai 416×320 detectors: the NNA requantizes; ReLU, depthwise 3×3, per-channel 1×1, 2×2 maxpool, ConvTranspose 2×2/2, concat, add; per-channel 1×1 chains folded into the conv before them) and the SiLU kind (Ingenic's yolov5n, 640×384 RGBA: 6×6 stem, SiLU requant, SPPF, nearest upsample, Normalize). `-e nna` pins to CPU0 and drives the real NNA; `-c` checks every engine conv against the CPU conv. It replaced `onnarun` (yolov5n with a text graph from `mkgraph.py`, and its loader `nnabin.c`), removed 2026-11-25 |
| `onnaq.h` | onnaq as a library (build `onnaq.c` with `-DONNAQ_LIB`): `onnaq_open(bin, graph, nna)`, `onnaq_run(rgb)`, `onnaq_head(k, …)`, `onnaq_anchors(stride, …)`, `onnaq_close()`; errors return -1 with `onnaq_error()` instead of exiting. One model per process. raptor-hal's onna backend (`ONNA=1`, `src/hal_ivs_onna.c`) vendors it. The graph may carry `anchors <stride> w0 h0 w1 h1 w2 h2` lines (`mkgraphq.py --anchors`). |
| `luttest.c` | the MXU3 256-entry byte table (`LUT256`: 4 × GSHUFVB) against a plain lookup; PASS under QEMU `-cpu XBurst2-MXU3.1` and on the v4 |
| `nnaeng.h`, `nnaeng.c` | the NNA conv driver: our own tiling (one row of 8 output pixels per MAC, input window into F RAM, W re-encoded per 32-oc block), the 6×6 stem as 6-row strips with two MACs, sums via CPOC + 16 NNDRD. Runs through a backend |
| `mxu3.h`, `be_mxu3.c` | the camera backend: MXU3/NN instructions as `.word` (encodings from QEMU's `mxu3.decode`), VWR word 0 for scalars, VPR 0 for data |
| `mxu3test.c` | self-test of the MXU3 data path (LAO/SAO, MFCPUW/MTCPUW); PASS under user-mode QEMU |
| `hwreplay.c` | runs a traced venus instruction stream on the real NNA and compares every read-back (silicon check of the backend) |
| `vrqtest.c` | the MXU3 SiLU requant (`nna_vrq_*`) against the scalar path for every SiLU table of a `.bin` (read with onnaq's loader: `onnaq.c` built in): all t in −40000..40000, then random sums with each layer's B and Fl. PASS under QEMU `-cpu XBurst2-MXU3.1` and on the v4 (369M cases) |
| `vaddtest.c` | the MXU3 residual add (`nna_vadd`) and concat rescale (`nna_vrescale`) against the scalar path for every input: all 65536 (a, b) pairs per add record, all 256 codes per zero-point-128 rescale factor of the graph (concats, Normalize; onnaq's loader and graph: `vaddtest model.bin`). PASS under QEMU and on the v4 (3.8M cases) |
| `vrqbench.c` | per-call cost of the MXU3 requant on the camera |
| `membench.c` | the v4's memory: streaming vector stores (with and without PrepareForStore) and reads |
| `rqbench.c` | per-element cost of the fused requant's stages on the camera (MXU3 t, clamp + store, LUT) |
| `nnstress.c`, `mxstress.c`, `setaff.c` | stability tests: `nnstress secs qsum mem` (NN instructions, with `mem` interleaved with MXU3 loads/stores: the T41 dual-core deadlock reproducer), `mxstress` (MXU3 loads/stores only), `setaff pid mask` (sched_setaffinity; the camera has no taskset). See `nna_pin_isp_cpu0()` in `be_mxu3.c` |
| `wfprobe.c` | F writes under a multi-word WFCW program, interleaved with MACs as a tiled conv does (marker pixels + identity weights); env `WFCW`, `R5B` (0x5b), `R21`, `PLANE`, `PITCH`, `TILES`. Shows that `0x5b = 1` changes where silicon puts F writes. Same code on oracle and silicon |
| `fprobe.c` | which F RAM address each output pixel reads (marker pixels + identity weights); same code on oracle and silicon |
| `v4run.sh` | runs `onnaq.mips -e nna -T` on a T41 camera **safely**: stops raptor's S31raptor watcher (it restarts rvd within ~5 s), then rvd (two processes on the NNA wedge the T41: watchdog reset), polls that rvd stays down, restores with `S31raptor restart` (also on failure). `SSHPASS` for a camera without a key |
| `../nnaop/m416/mgkcmp.sh` | onnaq `-e cpu` against Ingenic's PC executor (`mgkdump.cc` in the magik Docker image), head by head; graph from the `.bin` (`-`) or a `.qgraph` |
| `../nnaop/m416/bisect_onnaq.py` | per-layer bisect: `ONNAQ_DUMP` tensors vs the executor's, in graph order (`mkgraphq --names` maps ids to names); found the x64 requant shift |
| `oracle/` | host test oracle: `mkoracle.sh` extracts QEMU's NNA engine from your QEMU checkout at build time (GPL, never copied here); `replay.c` checks it against a venus trace; `be_oracle.c` (one level up) drives it |

The graph comes from the `.bin` itself (`-`). For the QSUM models `../nnaop/m416/mkgraphq.py model.onnx > model.qgraph`
still writes a text graph (it can carry anchors); the numbers always come from the `.bin`.

```sh
GOLDEN_DUMP=../gold python3 ../nnaop/golden.py model.bin model.onnx input_rgba.raw   # test vectors
make check          # host
make ORACLE=1 onnaq && ./onnaq -e oracle -c model.bin - input_rgba.raw out   # NNA driver vs oracle
make check-mips     # T41 build (xb v4) under user-mode QEMU -cpu XBurst2
make check-vrq check-vadd MODEL=yolov5n_t41.bin   # MXU3 requant / add / rescale vs the scalar path
```

Status, 2026-11-25 (onnaq, QSUM models):
- All four thingino-ai model types run on `onnaq`: persondet (v1, v2), YOLO-N, YOLOX-Nano and MobileNetV4
  + YOLOX neck. Heads byte-identical to Ingenic's PC executor (`mgkrun`), `-e cpu` and `-e oracle -c`.
- The `.bin` header is the converter's op list: length-prefixed names XORed with a 9-byte key
  (`NAMEKEY`). onnaq reads it for the add records (after each `Add/add_...` kernel) and each conv's
  requant shift (after its `Convolution/base_i8/...` kernel: `q = floor(((S << sh) + B) / T) + 128`;
  usually 2, 6 where the converter scaled the weights down). `mkgraphq.py --bin` matches convs to W
  blocks by weight and adds / concats by name (the converter's order is not the ONNX order).
- Every conv runs on the NNA: stride-2 layers with a bottom/right-only pad get a padded input copy;
  convs with shift != 2 read back their sums and requantize on the CPU. Residual adds use the MXU3
  `nna_vadd`, checked against the node's table on all 65536 input pairs at first use.
- **No graph file needed:** `onnaq model.bin - frame.rgb out` (or `onnaq_open(bin, NULL, nna)`) builds
  the graph from the `.bin`'s own op list: op types and tensor names from the header, each conv's
  kernel / stride / pads / channels / shift from its record, a ConvTranspose as UpSample (a CPU op,
  no "AIE" tag) + 2x2 conv, the input shape after `data_1`. Same heads as with mkgraphq's graph for all
  six model files. A `.qgraph` is still accepted (it can carry anchors, which the `.bin` does not).
- On the v4 (stock, no nmem), 20 runs each, heads identical to the PC executor: persondet v2 30.8 ms
  avg / 20.0 best, YOLO-N 79.6 / 64.7, YOLOX-Nano 67.7 / 54.8, MobileNetV4 + YOLOX neck 74.5 / 60.9.
  What still runs on the CPU (`-T`): unfolded per-channel 1x1s, concat rescales, nearest upsampling.
- Env switches for the T41 reset hunt (A/B, heads unchanged; checked with `-e oracle -c` on all four
  model types): `ONNA_NO_VADD=1` keeps residual adds on the scalar table, `ONNA_CPU_RAW=1` keeps the
  non-head convs with shift != 2 on the CPU, `ONNA_CPU_S2=1` keeps stride-2 / bottom-right-padded
  convs on the CPU. Per model (raw / stride-2 convs): persondet v2 0 / 0, YOLO-N 0 / 6,
  YOLOX-Nano 0 / 6, MobileNetV4 + YOLOX neck 3 / 6.

Status, 2026-11-25 (onnaq, yolov5n):
- **One runner:** `onnarun`, its loader `nnabin.c` / `nnabin.h`, the text `.graph` format and
  `nnaop/mkgraph.py` are removed. `struct nna_node` (the conv for `nna_conv`) moved to `nnaeng.h`.
  `vrqtest` and `vaddtest` read the `.bin` with onnaq's loader (`onnaq.c` built in; `vaddtest` needs
  no graph file now): yolov5n 57 SiLU tables / 8.2M cases, 3.8M add + rescale cases, PASS under QEMU.
- `onnaq` runs Ingenic's yolov5n `.bin` (640x384 RGBA, the one `onnarun` ran) from the `.bin` alone:
  `onnaq model.bin - input_rgba.raw out`. Heads **byte-identical to venus / onnarun** on both frames
  (COCO 322864 and the v4 night frame), `-e cpu` and `-e oracle -c`, host and MIPS (`-e cpu` under
  QEMU `-cpu XBurst2-MXU3.1`). Host x86 `-e cpu`: 1.3 s (onnarun: 5.4 s). Not yet run on the camera.
- What the SiLU kind adds to the op-list graph (the model kind is decided by the stem: a 6x6 conv
  with a SiLU table and output zero point 128):
  - SiLU convs: the record's SiLU table (the `04 00 00 00 20 00 00 00` marker, stored twice) gives
    onnarun's requant, `q = lut[rint(f32(S + B) * Fl)]`, fused into the NNA read-back as in onnarun.
    Pad value = the conv record's input zero point (field [1]: input in the low 16 bits, output in
    the high 16; 128, the stem's input 0).
  - The stem: a 1x1 record over 6 input chunks (chunk kx, lane 4 ky + c) run as 6x6/2 pad 2 on
    RGBA through nnaeng's stem strips.
  - Concats: a conv with several inputs is venus's implicit concat (identity); `Concat` ops carry
    their own record (`n`, zero points `-128` x n, `n`, `s_1..s_n`, `1/s_out`), applied as
    `rint((x - 128) a) + 128`. A record's W input chunks follow venus's layout (a concat of two
    16-channel tensors is two chunks of 16); onnaq maps them onto its plain tensors.
  - `Normalize` (`ptq_model_concat_fuse_...`): a rescale whose factor is in the type-30 record after
    the conv record before it. `Pooling` records give 2x2/2 or 5x5/1 pad 2 (SPPF); an `UpSample`
    with an "AIE" tag is a nearest x2 (without one it is still half of a ConvTranspose).
- The QSUM models are unchanged: heads identical to before for persondet v1/v2, YOLO-N and
  YOLOX-Nano on both 416x320 frames, `-e cpu` and `-e oracle`.

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

Licence: MIT, the same as ojzdl and raptor-hal's own sources (SPDX header in each source file); Copyright (C) 2026 Thingino Project. The test oracle (`oracle/`) is built from QEMU's GPL-2.0-or-later NNA engine at build time; that engine code is never copied into this repo, and the oracle binary it produces is GPL.
