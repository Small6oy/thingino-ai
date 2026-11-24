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
| Time per frame | 2.1 s (first run); **412 ms** after speed round 3 | ~80 ms |
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

### W RAM: 288 entries

A T41NQ's W RAM holds **288 entries (72 KB)**; writes past them wrap to entry 0. QEMU's model said
1024; the fix and a qtest are a QEMU PR. `tools/onna/wprobe.c` measures it: groups whose weights
encode their own number, read back through MACs that step the W pointer. venus never loads more than
288 entries at once and never writes the W read pointer (`0x30`) other than as 0. onna does the same:
at most 288 entries per load, the pointer stepped by MAC immediate bit 0.

### Speed round 3

From 825 ms to **412 ms min (433 avg)** on the stock v4, output still equal to venus's:
- requant on MXU3, 16 channels at a time (vector add, int→float32, multiply, float32→int with
  nearest-even). `rqvtest.c` compared it with the scalar path in 2,000,000 cases on silicon,
  250,000 of them exact ties: 0 mismatches;
- requant fused into the read-back, so no int32 sums buffers;
- interior tiles without bounds checks;
- every output block's weights kept loaded together (288-entry chunks), so F is written once per
  tile;
- a separable 5×5 maxpool.

### Speed rounds 4 and 5

From 412 ms to **145 ms min (163 avg)** on the stock v4, 15.2 MB peak RSS. The heads still equal
venus's silicon output on both test frames.

| Change | Forward, min |
|---|---|
| concat rescale through a 256-entry map; requant t for a whole tile at once; tensors reused | 302 ms |
| **the whole SiLU requant in MXU3**, no lookup table | 220 ms |
| the two 16-channel halves interleaved, read-backs straight into vector registers | 193 ms |
| heads fused the same way (vector add, int→float32, multiply) | 161 ms |
| stem: one column-packed strip per output row; each tile's F is a slice of it | **145 ms** |

The vector SiLU requant, per 16 channels, in the scalar reference's float32 order:
1. `t = rint(f32(S + B) * Fl)`, as in round 3.
2. `tc = clip(t, 0, 32767)` (vector max/min), segment `s = tc >> 10`, `fr = tc & 1023`.
3. `dy[s]` and `y[s]`: one GSHUFVB each. That's a byte gather from the 64-byte int16 table.
   - Each word's selector bytes are `0x80, 0x80, 2s, 2s+1`, built as
     `(s << 17) + (s << 25) + 0x01008080`.
   - That puts the entry in the high half; an arithmetic shift by 16 sign-extends it.
4. `q = rint((f32(y) + f32(dy) · f32(fr) · 2⁻¹⁰) · 2⁻⁷)`, clamped to 0..255. Multiplying by a
   power of two is exact, so this equals the reference's divisions.
5. A second GSHUFVB per half packs the bytes; one 32-byte store per pixel.

GSHUFVB is an MXU3.1 instruction: in QEMU use `-cpu XBurst2-MXU3.1`, because plain `XBurst2`
raises Reserved Instruction. The T41 has it. `vrqtest.c` checks the vector path against the scalar
one for every SiLU table of the model: every t in −40000..40000, plus random sums with each
layer's own B and Fl. Result: 41M cases under QEMU and **369M cases on the v4, 0 mismatches**.

Where the 145 ms goes now: stem ~20 ms, heads ~11, the other convs ~75, and add + concat + maxpool
on the CPU ~27.

### Speed round 6

From 145 ms to **96–100 ms min (113 avg)** on the stock v4. The heads still equal venus's
output on both frames, and every conv still equals the CPU conv on the oracle.

| Change | Forward, min |
|---|---|
| 16-channel blocks: two pixels per requant call; NN register writes inline | ~135 ms |
| 5×5 maxpool on MXU3 (byte max, 64 channels per op) | ~127 ms |
| residual add on MXU3 | ~118 ms |
| concat rescale on MXU3; all tensors 64-byte aligned | 111 ms |
| **F straight from the tensor**: two 32-byte loads (pixel x and x + 1 of a plane, C bytes apart) into the NNDWR register, no copy | **96–100 ms** |

- **The add on MXU3 is exact:**
  1. `num = 64(wA·a + wB·b) + B` in 32-bit lanes.
  2. `q = rint(f32(num) · f32(1/T))`.
  3. Correct q to the floor with `r = num − q·T`: subtract 1 when r < 0, add 1 when T ≤ r.
  4. +128, clamp.

  |num| < 2²⁴, so the conversion is exact.
- **The concat rescale** is `rint((x − 128)·a) + 128` in float32, as the scalar code.
- **`vaddtest.c` checks every input:** all 65,536 (a, b) pairs per add record, and all 256 codes per
  concat factor. 3.8M cases, 0 mismatches, under QEMU and on the v4.

**What didn't work:** staging every conv input into a padded, plane-major buffer. It made F writes
trivial but cost ~49 ms per frame, because it writes every activation a second time. Reading F in
place is what this core's memory bandwidth needs.

Left: the 16-channel inputs (16-byte pixels still go through a copy, ~15 ms), the concats (~8 ms),
and the stem's input strips (~6 ms).

### Speed round 7

From 96–100 ms to **85 ms min (101 avg)** on the stock v4, heads still equal to venus's.

- **16-channel tensors are stored with 32-byte pixels** (channels 16–31 zero), so the NNA reads
  every layer's input in place. Conv, add and concat take the pixel stride. The other ops get a
  dense copy, but none of them sees one in yolov5n.
- **The 16+16 concat** is two byte shuffles and an OR per two pixels.
- **Rescaled concat inputs** go straight into place.

**Not yet like for like:** venus's ~79 ms was measured with the HomeKit daemon stopped and its
memory carve-out in place. onna's 85 ms is on the stock camera with the streaming daemons sharing
the CPU.

### Speed round 8

From 85 ms to **78–80 ms min (94 avg)** on the stock v4, with the streaming daemons running. The
heads still equal venus's.

- **F windows.** A MAC reads F at `word + (r·Sy + ky)·pitch + c·Sx + kx`, where `word` is its
  operand. FX/FY only feed the padding check. So one F write per window of many 2×4 tiles is
  enough, with each MAC placed in it by its word. Windows up to a 508-pixel plane work on the T41.
- **A shorter requant chain** (27 → 22 vector ops per half):
  - the gathered table entries are used scaled by 2¹⁶, which doesn't change float32 rounding, so
    the last constant is 2⁻²³;
  - the selector is built by one multiply;
  - SATSU does the 0..255 clamp.

  `vrqtest`: 187M cases on silicon, 0 mismatches.
- **Border rows.** Only the pixel pairs at the border take the scalar path; rows outside the input
  send one padding vector.

**What the v4 showed about F:**
- NNDWR costs nothing measurable, but loading its 64 bytes does: cache misses, ~10 ms per frame.
- Software prefetch doesn't help.
- Fewer NNDWR alone gained nothing until the border rows were fixed.

### Speed round 9: venus's speed

- the requant's shared vector registers loaded once per conv;
- 13 concat inputs written in place by the convs that produce them;
- the 6×6 stem's input strip keeps input row y in slot y mod 6, so each output row writes only its
  2 new rows, with 3 weight variants for the slot rotation.

`onnarun -n` now also checks that every run gives the same heads.

**Like for like:** the same Wyze Cam v4 boot, set up for venus (nmem, 128K L2, HomeKit daemon
stopped), runs interleaved, 20 frames each:

| | avg | best |
|---|---|---|
| venus 0.3.7.1 | 81.6 / 82.0 ms | 69.1 / 68.5 ms |
| **onna** | **83.0 / 82.9 ms** | **69.6 / 69.9 ms** |
| venus 0.1.2.5 | 84.7 / 86.0 ms | 70.1 / 68.9 ms |
| **onna** | **89.0 / 86.9 ms** | **70.6 / 72.0 ms** |

onna runs at venus's speed within run-to-run noise, with byte-identical output. It needs no
boot-time memory reservation, no L2 change and no vendor libraries.

**Ruled out:**
- venus's DMA staging of F, which needs physically contiguous memory and the 128K-L2 ORAM;
- PrepareForStore (stores already stream at ~2.5 GB/s);
- 16-byte F pixels (unproven on silicon: venus never uses them).

### Side by side with venus on the same camera

A Wyze Cam v4 set up for venus (`nmem=20M`, 128K L2), with all three runtimes on the same frame in
the same boot:

| | onna | venus 0.1.2.5 | venus 0.3.7.1 |
|---|---|---|---|
| Output (heads) | = venus | reference | identical to 0.1.2.5 |
| Run, avg (min) | 913 ms (869) | 78.8 ms (69.6) | 79.8 ms (69.4) |
| Process RSS | 18.0 MB | 5.9 MB | 8.6 MB |
| `nmem` | none | 13.6 MB used, 20 MiB reserved at boot | 13.7 MB used, 20 MiB reserved |
| Size | 142 KB static | 4.5 MB + libstdc++ | 11.2 MB + libstdc++ |

On silicon, `tools/onna/nnabench.c` puts each NN instruction at a few nanoseconds. A whole tile
(MAC, CPOC, 16 read-backs) takes 0.17 µs. So onna's remaining time is all CPU work: requant and
packing F.

Upstream QEMU modelled the slicing in `ae58bace25`. Its model reproduces the silicon probe
address for address. Writing F with no WFCW program (`0x5a = 0`) still differs from silicon:
[gtxaspec/qemu#63](https://github.com/gtxaspec/qemu/issues/63).

Also needed on silicon, as venus does it:
- the F write program (WFCW) set explicitly, even when it's the default;
- NNCMD 0x60 before the F writes;
- 2×4 tiles (`0x38 = 0x12`) for normal convs, and the one-row mode (`0x38 = 0x3`, with
  `0x21`/`0x23 = 3`) only for the 6×6 stem.

## A second kind of model: QSUM (the person/pet/package detector)

thingino-ai's 416×320 person/pet/package detector (2 heads; depthwise 3×3, per-channel 1×1, ReLU, 2×2 maxpool, ConvTranspose, one concat) runs through onna's second runner, `tools/onna/onnaq.c`. On a stock Wyze Cam v4 (no nmem, 256K L2) its heads are **byte-identical to venus's** on two frames, and every engine conv equals the CPU conv on silicon (`-c`).

**How venus runs this kind of model** (trace of venus 0.1.2.5, 60 engine segments):
- **One NNA segment per Conv or ConvTranspose. The NNA requantizes every conv except the heads:** QSUM through BT, `q = clamp(floor((4 S + B) / T) + 128)`, read back as bytes with NNDRD 0x20.
  - ReLU is folded into B/T: the zero point is 0, so the clamp at 0 is the ReLU.
  - Padding is 0.
  - The heads read raw sums: `f32(S + B) · Fl`.
- **W headers:** `[bytes, 7, nob, nic, kh, kw, 4, 32, icg]`.
  - `icg = 32`: dense, stored as streamed.
  - `icg = 1`: depthwise or per-channel. 32 bytes per block and tap (4 slices × 8 B); venus expands them to diagonal 32×32 groups in snake tap order.
  - The param record follows: B at +0x25, T (heads: Fl) at +0xa5, 256 B per 32-oc block.
- **Stem (3×3/2, RGB):** a 1×1 conv on 32-byte pixels, 27 of whose 32 slots hold (c, ky, kx) of the window.
- **Per-channel 1×1 pairs** (weight −1, the same B/T in every channel) form a clip; each pair is one byte table per layer.
- **ConvTranspose 2×2/2:** a 2×2 conv over the dilated input, with the kernel stored flipped.
- **Concat:** a header record `n, 0 × n, n, s_1..s_n, 1/s_out`; input i becomes `rint(x · s_i / s_out)`.

**Golden model:** `tools/nnaop/m416/golden416.py` (the `.bin` plus the ONNX structure). It is byte-identical to Ingenic's PC executor, which equals the v4 silicon, on 7 frames. `mgkdump.cc` dumps the executor's intermediate tensors by name.

**QEMU 0.7.26 does not reproduce venus on this model:**
- The heads differ from silicon.
- The stem goes wrong in half its output columns.
- The first depthwise layer differs even where its inputs are right.

The likely cause is venus's F layouts here (a two-word WFCW program in the stem). The C runner is checked against the golden model, the host oracle and silicon.

**onnaq:**
- **The engine (`nnaeng.c`)** gains a QSUM read-back:
  - BT is written after the W load and NNCMD 0x60, from row 0 (`0x5e`);
  - per tile: `0x42 = 4·block`, `NNCMD 0x84`, `NNCMD 0x8b`, then 4 × NNDRD 0x20.
  
  **Silicon needs venus's order:** BT before the W, with RDBT as `0x04`, gives zero bytes. QEMU takes both.
- **Depthwise:** each 32-oc block MACs only its own plane.
- **CPU:**
  - per-channel 1×1 chains are folded into the conv before them as one byte table, applied to the QSUM bytes with MXU3 (`LUT256`: 4 × GSHUFVB; `luttest.c`);
  - the 2×2 maxpool uses MAXU.B;
  - the deconv and concat are byte tables.

**Speed on the stock v4: 23.1 ms best / 33–35 avg per frame, faster than venus** (28.7 / 37 ms, measured on venus's own boot with nmem and the 128K L2). The first silicon run took 2.5 s. The steps, each one exact:
- tensor buffers kept across runs;
- the clip pairs folded into the convs;
- the maxpool and the table on MXU3;
- 16-channel tensors stored as 32-byte pixels, so F is written straight from the tensor;
- an unrolled stem im2col;
- the deconv folded into the concat;
- the QSUM read-back stored straight to the tensor with SAO.

**Why QEMU differs:** register `0x5b`. venus sets it to 1 in the stem; on silicon it changes where F writes land, and the model ignores it (`tools/onna/wfprobe.c`, gtxaspec/qemu#65).

## What's left

1. **Speed: done for yolov5n, at venus's speed** (same boot, same output). The person/pet/package
   detector runs (above) at 23.1 ms against venus's 28.7.
2. **The rest of the header:** a decoded op list instead of rules. The name strings are
   obfuscated. Record types seen: type 35 after a conv feeding a header-record concat, type 30
   after a passthrough-concat producer, type 1 otherwise.
3. **Other models:** int4 weights (msf_mode and slice count change); dense k×k convs in the QSUM
   kind (onnaq takes dense 1×1 and depthwise k×k only).
4. **The AIP preprocessing** (NV12 → letterboxed RGBA) is a separate hardware block, not looked at
   here.
