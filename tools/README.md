# tools

| Folder | What |
|---|---|
| [jzdl/](jzdl/README.md) | T31 JZDL model toolkit: `.bin` parser/writer, bit-exact NumPy reference runtime, loader check, probe, camera/QEMU runner (`jzdlrun`). Format notes: [docs/jzdl-format.md](../docs/jzdl-format.md). |
| [ojzdl/](ojzdl/README.md) | Our own JZDL runtime for the T31: C99 + MXU2 kernels, bit-identical to libjzdl and as fast. Notes: [docs/jzdl-runtime.md](../docs/jzdl-runtime.md). |
| [aitest/](aitest/) | on-camera benchmark harness: grabs a frame from rvd at a fixed rate, runs a backend (JZDL or ncnn), reports per-phase ms, CPU, RSS and detections, can save frames. `build.sh` cross-builds it. |
| [nna/](nna/README.md) | T41 NNA (venus) runners `nnatest` / `nnappp`, their build, and the `t41l2` L2/ORAM module. |
| [nnaop/](nnaop/README.md) | NumPy model of venus on the T41 NNA (toward an open runtime), checked against a QEMU trace. Notes: [docs/nna-runtime.md](../docs/nna-runtime.md). |
| [onna/](onna/README.md) | open runtime for the T41 NNA in C (venus-compatible `.bin`): on a Wyze Cam v4 its output equals venus's, with no nmem and no soc-nna. Notes: [docs/nna-runtime.md](../docs/nna-runtime.md). |

The training pipeline and both exporters are in [train/](../train/). Ingenic runtime libraries,
model templates and toolchain libraries are never committed: the tools take them by path.

## conform.py: is a model the same everywhere it runs?

`python3 tools/conform.py MODEL_DIR [--frames DIR] [--update-card]`: for a model library folder or a
build_model.py output. T31: checkpoint re-export == model.bin, jzdl_ref == ojzdl, and (with the
offline QEMU runtime) libjzdl == ojzdl via `ojzdl/gate.sh`. T41: onnaq with the graph from the .bin,
the NNA driver vs the CPU (`-e oracle -c`), and (with Docker + mgkdump) Ingenic's PC executor.
Each check says PASS / FAIL / SKIP (why); exit 1 on a FAIL. `--update-card` writes the golden
output md5s into model.json. Run it before every model library PR.

## backendcmp.py: does the runtime or the converter cost detections?

`backendcmp.py frames --data DATASET OUT` picks validation frames with people (as training
preprocesses them, plus one `frames.u8` for mgkrun); `backendcmp.py onnaq MODEL.bin OUT/onna` runs
onnaq on them; `backendcmp.py score --data DATASET OUT SRC...` scores onnaq folders and mgkrun output
prefixes the same way: AP50 per class, and for frames with one or two clear people, people found
and false person boxes per frame at thresholds 0.10 to 0.40. The docstring shows how to build
`train/t41/mgkrun.cc` against another Magik PC executor (e.g. a Magik 3.x kit's pc-inference).
Results: [docs/one-model-two-exports.md](../docs/one-model-two-exports.md#same-t41-model-on-three-paths-2026-11-25).
