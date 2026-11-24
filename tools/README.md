# tools

| Folder | What |
|---|---|
| [jzdl/](jzdl/README.md) | T31 JZDL model toolkit: `.bin` parser/writer, bit-exact NumPy reference runtime, loader check, probe, camera/QEMU runner (`jzdlrun`). Format notes: [docs/jzdl-format.md](../docs/jzdl-format.md). |
| [ojzdl/](ojzdl/README.md) | Our own JZDL runtime for the T31: C99 + MXU2 kernels, bit-identical to libjzdl and as fast. Notes: [docs/jzdl-runtime.md](../docs/jzdl-runtime.md). |
| [aitest/](aitest/) | on-camera benchmark harness: grabs a frame from rvd at a fixed rate, runs a backend (JZDL or ncnn), reports per-phase ms, CPU, RSS and detections, can save frames. `build.sh` cross-builds it. |
| [nna/](nna/README.md) | T41 NNA (venus) runners `nnatest` / `nnappp`, their build, and the `t41l2` L2/ORAM module. |
| [nnaop/](nnaop/README.md) | NumPy model of venus on the T41 NNA (toward an open runtime), checked against a QEMU trace. Notes: [docs/nna-runtime.md](../docs/nna-runtime.md). |

The training pipeline and both exporters are in [train/](../train/). Ingenic runtime libraries,
model templates and toolchain libraries are never committed: the tools take them by path.
