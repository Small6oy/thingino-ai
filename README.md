# thingino-ai

On-camera AI detection for [Thingino](https://thingino.com/) cameras — **person, pet, package and sound**, detected locally and turned into HomeKit events, on-screen boxes and MQTT, with a web UI to switch each one on or off. No cloud. Open architectures, trained on open data.

This repo is the single home for the AI work: the tools, the models, the training pipeline, the design, and the camera integration. The integration code is developed on the upstream Thingino/raptor forks (so it can be contributed back) and **mirrored** here as patches — see [camera/BRANCHES.md](camera/BRANCHES.md).

## Detectors

| Modality | Classes | Method |
|---|---|---|
| Vision | person, pet, package | object detector on 360p frames, ~1 fps, motion-gated |
| Sound | baby crying, dog barking, coughing, alarm | small audio classifier on the mic PCM |
| Classic CV | motion, lens-cover / tamper | frame diff / threshold, no model |

## Two hardware paths

| | Wyze v3 (T31) | Wyze v4 (T41) |
|---|---|---|
| Runtime | JZDL on the CPU | venus on the NNA |
| Model built with | `jzdl` | `magikconv` |
| SIMD / accel | single core, MXU2 | MXU3 + NNA |

One model is trained once and exported two ways: a bit-exact JZDL `.bin` for the T31, and a float ONNX that Ingenic's converter turns into an NNA `.bin` for the T41. See [docs/one-model-two-exports.md](docs/one-model-two-exports.md); [train/](train/) builds both with one command.

## Pipeline

```
frame / audio → backend (JZDL · NNA · audio CNN) → NMS + tracking gate
   → event {class, box, score} → rmd (per-class region, mask, cooldown)
      → split, per class and independent:  notify (MQTT)  ·  record (HomeKit Secure Video)
         → rhkd HomeKit sensors  +  on-screen boxes
```

Two rules borrowed from a shipping commercial camera on the same T31 silicon (the Aqara G100): a per-object **tracking gate** before any report, and **detect / notify / record** as three independent per-class decisions.

## Layout

```
tools/        aitest (benchmark harness) · jzdl (T31 toolkit) · nnatest (NNA harness) · magikconv (NNA exporter)
train/        the shared QAT/dataset pipeline → one ONNX, exported to both backends
models/       trained + converted models (Git LFS) — person/pet/package, labels, sound
camera/       thingino integration, MIRRORED from the forks: patches/ + BRANCHES.md + webui/
docs/         design: ai.md, tasks.md, benchmark.md, jzdl-format.md, the G100 detection reference
data/         benchmark / calibration frame + audio sets (Git LFS)
```

## Tools

| Tool | What |
|---|---|
| `aitest` | on-camera benchmark harness: grab a frame, run a backend, report per-phase ms + CPU/RSS/system/memory; motion-gating; saved event frames |
| `jzdl` | T31 (JZDL) model toolkit: parse/write/convert (byte-exact), a NumPy reference runner (bit-exact), a camera/QEMU runner, and the exporter (trained model → JZDL `.bin`) |
| `nnatest` | T41 NNA (venus) inference harness |
| `magikconv` | wrapper around the vendor magik-transform-tools: ONNX + cfg → T41 magik `.bin`, with calibration frames and a host-side accuracy check |
| `train` | the shared QAT / dataset pipeline that produces the one ONNX both exporters consume |

## Public-safe

This repo has a public mirror. It contains **no IP addresses, host names, local paths or household details** — those stay in private notes. A pre-commit check rejects them.

It also contains **no Ingenic vendor binaries** (runtime libraries or vendor model templates) — the tools reference them by path, and the guard rejects them. Templates for training/conversion are supplied by env/arg, not vendored.

## Status

Early. The design and the T31 model format + converter are done (converter is bit-exact against the real camera); training and camera integration are in progress. Roadmap and gates: [docs/tasks.md](docs/tasks.md). v3 first.

## License

TBD — intended open; models trained on open data only.
