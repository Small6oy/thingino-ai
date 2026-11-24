# nnaop: a NumPy model of venus on the T41 NNA, checked against a QEMU trace

Scripts behind [docs/nna-runtime.md](../../docs/nna-runtime.md). Each one takes a working directory
holding one traced frame and its inputs. None of these files are in this repo: they come from your
own run, and the trace contains your model's weights.

| File | From |
|---|---|
| `nn.bin`, `trace.log` | QEMU `ingenic-0.7.26`: `-global ingenic-nna.log=nn.bin -trace enable=ingenic_nna_write -trace enable=ingenic_nna_op -D trace.log`, one inference with `tools/nna/nnatest` |
| `yolov5n_t41.bin` | the venus model (magik-transform-tools 2.0.4) |
| `yolov5n.onnx` | the ONNX the model was converted from (to identify layers) |
| `input_rgba.raw`, `out0/1/2.f32` | `NNATEST_DUMP=dir` from the same run (the AIP input and the head outputs) |

Python 3 with numpy and onnx.

| Script | What it does |
|---|---|
| `nnaop.py <dir>` | names every command type; per-layer table; NNDMA totals. Writes `resets.npy` and `dma.npy` (run it first) |
| `wmap.py <dir> <model.bin> <conv_q.npy> [seg]` | one layer's W RAM → 32×32 weight groups, matched to an ONNX conv; finds the stream in the `.bin` |
| `fmap.py <dir> <seg>` | replays QEMU's F-write / WFCW / MAC addressing for a layer. Records each F pixel's input position, the W entry/plane/tap of every MAC, the tile of every CPOC, and every product accumulated per output pixel |
| `convref.py <dir> <seg> [onnx conv]` | NNA side of a conv: decodes W from the `.bin`, matches it to ONNX, recomputes every traced sum (and, for absolutely tiled 3×3 layers, compares by position with a plain NumPy conv) |
| `ref01.py <dir>` | layer 0 end to end from the image and the `.bin`: NNA sums by position, then the CPU requant against every byte layer 1 receives |
| `rqall.py <dir>` | CPU requant + SiLU of every conv (and the concat rescales), checked against the inputs of later layers |
| `addref.py <dir>` | the residual adds (NNA QSUM) against their own operands |
| `head.py <dir>` | the head convs' float output against `out0/1/2.f32` |
| `binwalk.py <model.bin> <model.onnx>` | reads the `.bin` without a trace: every conv's weights, requant record and SiLU table, the add records and concat factors |
| `golden.py <model.bin> <model.onnx> <input_rgba.raw> [out0 out1 out2]` | **golden model**: runs the network as venus does from the `.bin` alone, and compares the heads with venus's outputs by position |

`golden.py` needs only the `.bin`, the ONNX and one AIP input (plus venus's outputs to compare
against). The trace checks: `nnaop.py`, then `fmap.py` + `convref.py` for each segment, then `rqall.py`, `addref.py`
and `head.py`. The `.bin` offsets in `ref01.py` are those of our yolov5n conversion. The other
scripts find blocks by searching for the traced W stream.
