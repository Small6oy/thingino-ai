# docs

Design and roadmap for Thingino on-camera AI detection.

- [ai.md](ai.md) — architecture, the two hardware paths, the pipeline.
- [tasks.md](tasks.md) — phased task breakdown with gates (v3 first).
- [benchmark.md](benchmark.md) — how we benchmark against the Aqara G100.
- [jzdl-runtime.md](jzdl-runtime.md) — ojzdl, our own T31 JZDL runtime: bit-identical to libjzdl, MXU2 kernels, as fast on the camera.
- [nna-runtime.md](nna-runtime.md) — toward an open T41 NNA runtime: what venus does, layer by layer, reproduced exactly in NumPy.
- [aqara-g100-detections.md](aqara-g100-detections.md) — the G100 detection/event catalog (the commercial reference we borrow from).

Household/LAN operational detail (camera addresses, build host) is kept in private notes, not here. The JZDL model-format notes migrate here next (held while a training run uses them).
