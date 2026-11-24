# Toward an open T41 NNA runtime

Goal: run detection models on the T41's NNA without Ingenic's closed runtimes. Those are venus
0.1.2.5 / 0.3.7.1 and Uranus (Magik 3.x).

venus 0.3.7.1 can ship. It's in ingenic-lib, and our models run on it byte-identically. What an
open runtime gains:
- **size:** libvenus 0.3.7.1 is 11.2 MB stripped, on cameras with ~28 MB free;
- **headers:** none are published;
- **control:** over memory use and scheduling.

Method: the one that cracked JZDL ([jzdl-format.md](jzdl-format.md)):
1. A NumPy reference first.
2. Checked bit for bit against the closed library, run under QEMU.
3. Then on the camera.

Tools: [tools/nnaop/](../tools/nnaop/README.md).

**Status:** a NumPy golden model reads a venus `.bin` (no trace needed) and runs it the way venus
and the NNA do. Its YOLOv5n head outputs are **bit-identical to venus's** on two frames, one of
them checked against a Wyze Cam v4. Next is a C driver with a tiling plan of our own. We don't
need venus's DMA plan, only its arithmetic, and the golden model has all of it.

| Part | Result |
|---|---|
| NNA multiply-accumulate, all 60 convs | bit-exact |
| CPU requant + SiLU, all 57 SiLU convs | exact |
| Concat rescale, upsample, SPPF concat | exact |
| Residual adds (on the NNA) | bit-exact |
| Detection-head float output | identical to the camera's output |
| SPPF maxpools, upsample, every layer in image coordinates | exact (golden model) |
| `.bin` reader (no trace) | convs, adds, concat factors: enough for YOLOv5n |
| C runtime `tools/onna` | **on a Wyze Cam v4: venus's output exactly**, with no nmem and no soc-nna; 2.1 s per frame (v1 tiling, not yet optimized) |

## The trace

[gtxaspec/qemu](https://github.com/gtxaspec/qemu)'s `ingenic-nna` model reproduces the T41NQ's NNA
byte for byte on our models (release `ingenic-0.7.26`). Its source,
`hw/misc/ingenic-nna.c`, is the most precise spec of the unit there is. The release can already
log everything, without a rebuild:

```
-global ingenic-nna.log=nn.bin                                  # every NN instruction:
                                                                #   <u32 op><u32 imm><64-byte vector>
-trace enable=ingenic_nna_write -trace enable=ingenic_nna_op -D trace.log
                                                                # NNDMA descriptors and channel
                                                                # starts, in order with the NN ops
```

One frame of YOLOv5n (640×384, venus 0.1.2.5, model from magik-transform-tools 2.0.4) is
1,297,913 NN instructions and 2,201 NNDMA chains. Tracing doesn't change the result: the outputs
are byte-identical to an untraced run and to the camera.

The NNA is **not a command queue**. CPU0 drives it with MXU3's NN instructions, and feeds it
through its own DMA:

| Instruction | Use (yolov5n) |
|---|---|
| NNRWR | 33 engine registers: tile position, map size, W read pointer, planes, tap mask, data formats, msf_mode, F write line, the WFCW (F write) program |
| NNDWR | W RAM quarters (0x00/0x02/0x04/0x07), F through ports 0/1 (0x20/0x21), BT (0x40) |
| NNMAC | multiply-accumulate of 8 pixels × the taps × the planes |
| NNCMD | 0 = reset (once per layer); bits 0–3 = CPOC / CLROC / RDBT / QSUM; bits 7:5 = tile moves |
| NNDRD | the raw 8 px × 32 oc sums (0x00–0x0f), or the QSUM output (0x20) |
| NNDMA | DDR→ORAM on +0x8000/+0x8004, ORAM→DDR on +0x8010 |

Open details that QEMU treats as no-ops, and still matches silicon:
- NNCMD tile moves 3 and 4;
- NNMAC bit 1;
- eight registers QEMU stores but doesn't interpret.

## The model as the NNA runs it

YOLOv5n is **67 engine layers: 60 convs + 7 residual adds**. Everything is 8-bit: 8-bit W and
8-bit features (uint8, zero point 128). Padding is 128.

**Weights.** The `.bin` stores them already in the NNA's W-entry layout. A layer's block is
copied 64-byte aligned into DDR, DMA'd to ORAM, and streamed with NNDWR.
- **An entry** is 256 B: `[32 oc][32 ic × 2 bit]`, LSB first.
- **A weight** of 8 bits is 4 entries (2-bit slices). msf_mode 2 gives `w = v − 128`.
- **A group** of 4 entries is a 32 × 32 matrix for one kernel tap of one 32-oc block. The `.bin`
  stores a 3×3 conv's groups in raster tap order; the engine streams them in its own order.
- **The stem** (6×6, stride 2, 3 → 16) is six groups. Group = kx; input channel = 4·ky + c over
  a "pixel" that is a vertical strip of 6 RGBA rows.
- **All 60 convs' weights match their ONNX conv within ±1** (the converter's rounding).

**Concat is free on the NNA.** A concatenated input arrives as separate 32-byte input planes, or
as two 16-channel halves of one plane.

**The NNA only accumulates.** Every conv reads its raw int32 sums back (NNDRD after CPOC). The
CPU applies the activation and requantizes.

### CPU requant + SiLU (every SiLU conv)

Each layer's W block in the `.bin` is followed by its parameters:

```
wend = end of the layer's W block
for each 32-oc block k:
    B_k  = int32[32]    at wend + 0x25 + 256·k
    Fl_k = float32[32]  at wend + 0xa5 + 256·k
SiLU table = 128 bytes after the marker 04 00 00 00 20 00 00 00 (stored twice, identical):
    32 segments, int16, interleaved: (dy_i, dy_16+i) × 16, then (y_i, y_16+i) × 16
```
```
t   = rint(float32(S + B) · Fl)                  # S = the NNA's int32 sum
seg = clip(t >> 10, 0, 31)                       # 1024 t-units per segment
y   = y[seg] + dy[seg] · (clip(t, 0, 32767) − 1024·seg) / 1024      # float32; t < 0 → y[0]
q   = clip(rint(y / 128), 0, 255)                # round half to even; 16384/128 = 128 = zero point
```
- The table is SiLU in fixed point: y = 16384 + SiLU(z)·128/s_out. It saturates at
  32640 = 255·128.
- B is the bias plus the table's offset, both in sum units.
- **Use float32.** float64 interpolation flips 11 of 2.4M values at exact ties.

### Concat rescale (CPU)

Inputs of a concat are brought to the concat's scale as `F = rint((q − 128)·a) + 128`. Upsample
(nearest) is a byte copy. The factor a is stored in one of two places:
- **a header record:** `n, −128 × n, n, s_1 … s_n, 1/s_out`, so a = s_i / s_out;
- **for a 2-input concat whose other input passes through unchanged:** a per-channel float in a
  record (type 30) after the tables of the conv that produces that other input.

### Residual adds (NNA)

A C3 shortcut runs as a 2-tap conv:
- taps (0,0) and (1,0), over an input whose rows interleave the two operands;
- diagonal unsigned weights w_A and w_B (msf_mode 0);
- the NNA's own requant (QSUM, shift 6), with one B/T for all channels:
```
out = clip(floor((64·(w_A·a + w_B·b) + B) / T) + 128, 0, 255),    B = T/2 − 8192·(w_A + w_B)
```
One header record per add: `2, w_A, w_B (int32), 2, s_A, s_B, B, T (float32)`, with
w_B = 255 and w_A = rint(255·s_A/s_B).

### Detection heads

The three output convs skip SiLU. They use the same B/Fl record, and output
`float32(S + B) · Fl` (not `S·Fl + B·Fl`).

## Reading the `.bin` without a trace

[tools/nnaop/binwalk.py](../tools/nnaop/binwalk.py):
- **Each conv** has a parameter record at the end of its W block: `02 00 00 00 02 00 00 00 01 00…`,
  with count 32 at +0x21. The W block is preceded by int32
  `[bytes, 7, oc-blocks, 1, kh, kw, 4, 32, 32]`. The 6×6 stem is stored as 1×1 with 24 entries.
- **Weight groups** go oc block → input chunk (32 channels; members of a concatenated input are
  chunked separately) → taps in raster order.
- **Adds:** the header add records, in network order. Operand A (w_A) is the shortcut.
- **Concats:**
  - inside one block (C3, SPPF): no rescale;
  - across blocks: the header concat records, in order;
  - or the per-channel factor after the passthrough input's conv.

  For YOLOv5n this matches the trace exactly. It's a rule, not yet a decoded op list.

Layers are identified against the ONNX graph, which also supplies the network structure.

[tools/nnaop/golden.py](../tools/nnaop/golden.py) runs the network from the `.bin`, the ONNX and the
AIP image. It compares the heads with venus's outputs by position. Two details that matter:
- the stem pads the raw image with 0, every other layer with 128;
- the concat rescale adds the zero point back after rounding.

## The C runtime: onna

[tools/onna/](../tools/onna/README.md) runs a venus `.bin` with a graph that `mkgraph.py` writes from
the ONNX file.
- **CPU ops** (`nnacpu.c`): requant + SiLU, head, concat rescale, add, maxpool, upsample. They're bit-exact
  against the golden model's per-op dumps, on the host and as MIPS under QEMU.
- **Loader** (`nnabin.c`): validates everything it reads from the `.bin`.
- **NNA driver** (`nnaeng.c`): uses **its own tiling**:
  - one row of 8 output pixels per MAC;
  - the input window written into F RAM, with padding from the engine's bounds check;
  - weights re-encoded per 32-oc block;
  - the 6×6 stem as 6-row strips with two MACs;
  - the sums read back after CPOC.

  It runs through a backend:
  - **on the camera:** `.word` NN/MXU3 instructions (`mxu3.h`, `be_mxu3.c`);
  - **on the host:** QEMU's own NNA engine as a test oracle, extracted from your QEMU checkout at
    build time by `oracle/mkoracle.sh` (it's GPL and isn't copied into this repo). Replaying
    venus's traced 1.3M instructions through it reproduces all 661,920 read-backs.

Results on yolov5n:
- **Host:** every conv through the driver equals the CPU conv (60/60), and the heads equal venus's.
- **System emulator:** gtxaspec QEMU 0.7.26, a Thingino v4 image. `onnarun -e nna` with the real
  instructions gives venus's output byte for byte on two frames.
- **It doesn't open `/dev/soc-nna` and needs no `nmem`:** the CPU feeds the engine directly.

### On silicon (Wyze Cam v4)

On a stock Thingino v4 (no `nmem`, 256K L2, `soc_nna` loaded but unused, video daemons running),
`onnarun -e nna` reproduces venus's output exactly on both frames.

| | onna (v1 tiling) | venus 0.1.2.5 |
|---|---|---|
| Time per frame | 2.1 s | ~80 ms |
| Peak RSS | 12.5 MB | 5.8 MB, plus 13 MiB of `nmem` reserved at boot |
| Needs `nmem`, a 128K L2 or `soc-nna` | no | yes |
| Size | 142 KB static binary | 4.5 MB of libraries |

`hwreplay.c` replays a traced venus instruction stream on the camera through the same backend:
every read-back matches. That's how we knew the NN unit needs no driver setup.

**The T41's F RAM is sliced.** QEMU models this only for the T40.
- A byte written to F RAM is stored as four 2-bit slices at lines l, l + d, l + 2d, l + 3d, with
  d = `0x37`/4, and a read sums them.
- QEMU's T41 model stores whole bytes. That's equivalent whenever nothing collides, and nothing
  does in any layout venus uses.
- **The rule for T41 code:** each plane's data must fit in its first `0x37`/4 lines; planes sit
  `0x37` lines apart; each plane is written separately.
- `fprobe.c` shows this directly: on silicon, each output pixel reads a sum of several F
  addresses; under QEMU, exactly one.

Also needed on silicon, as venus does it:
- the F write program (WFCW) set explicitly, even when it's the default;
- NNCMD 0x60 before the F writes;
- 2×4 tiles (`0x38 = 0x12`) for normal convs, and the one-row mode (`0x38 = 0x3`, with
  `0x21`/`0x23 = 3`) only for the 6×6 stem.

## What's left

1. **Speed,** from 2.1 s toward venus's ~80 ms:
   - write F once per row band, not per tile;
   - stream F with WFCW programs;
   - pipeline the next MAC before the current read-back;
   - MXU3 for the CPU ops;
   - run the residual adds through the NNA's QSUM.
2. **The rest of the header:** a decoded op list instead of rules. The name strings are
   obfuscated. Record types seen: type 35 after a conv feeding a header-record concat, type 30
   after a passthrough-concat producer, type 1 otherwise.
3. **Other models:** the person/pet/package detector, and int4 weights (msf_mode and slice count
   change).
4. **The AIP preprocessing** (NV12 → letterboxed RGBA) is a separate hardware block, not looked at
   here.
