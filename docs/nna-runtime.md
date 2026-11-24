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

**Status:** a NumPy model of everything venus computes for a YOLOv5n reproduces one traced frame
exactly. Its final output is identical to the one a Wyze Cam v4 produced for the same frame.
What's left is the data movement: the DMA/tiling plan. Then come a header walker and a C driver.

| Part | Result |
|---|---|
| NNA multiply-accumulate, all 60 convs | bit-exact |
| CPU requant + SiLU, all 57 SiLU convs | exact |
| Concat rescale, upsample, SPPF concat | exact |
| Residual adds (on the NNA) | bit-exact |
| Detection-head float output | identical to the camera's output |
| Layout / DMA plan, SPPF maxpools | open |
| `.bin` header walker | partial: record layouts known, offsets still found by search |

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

## What's left

1. **Layout / DMA plan.** Most 1×1 layers, and some 3×3, feed the engine 2×4-pixel tiles in
   tile-local coordinates. So the image position of a result exists only in the DDR addresses of
   the NNDMA chains. That is checked "as a set" for now: every requantized vector of every layer
   appears in a later layer's input. The SPPF maxpools need positions to be checked.
2. **A `.bin` header walker,** to replace the offset searches. The record types seen so far:
   - per-conv parameters;
   - SiLU tables;
   - concat (type 35 after a conv, plus header records);
   - add records;
   - the record that ends each SiLU table (type 30 or 1).
   Name strings in the header are obfuscated.
3. **A C driver:** issue the NN instructions and NNDMA for a parsed model, and do the CPU ops
   (requant/SiLU, concat rescale, maxpool, upsample).
4. **The AIP preprocessing** (NV12 → letterboxed RGBA) is a separate hardware block, not looked at
   here.
