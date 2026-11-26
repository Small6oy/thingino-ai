# On-camera AI detection on Thingino (v3 + v4)

Goal: run local detection on our cameras and turn it into HomeKit events, OSD boxes and MQTT, with a web UI to switch each one on or off. No cloud, no vendor models, open architectures trained on open data. Three kinds of detector:

- **Vision** (video frames): person, pet, package to start. A commercial example on the same silicon is the Aqara G100 (T31, CPU only); we benchmark against it.
- **Sound** (audio): baby crying, dog barking, coughing, alarm — a small audio classifier on the CPU fed from the `rad` audio path, independent of the video detector. The G100 has this too (`CVoiceAlgo`), though no sound model actually ships on it.
- **Classic CV** (no model): motion and lens-cover / tamper — cheap, works on any SoC.

This folder is the umbrella for the whole thing. The deep technical sub-areas keep their own folders and this links to them:

- **[nna-runtime.md](nna-runtime.md)** — T41 NNA runtime (venus and our open replacement), model conversion, the `rvd` NNA backend. Already works on the v4 (yolov5n ~80 ms).
- **[jzdl-format.md](jzdl-format.md)** — our own JZDL model writer for the T31 CPU path. The v3 has no NNA, so JZDL on the CPU is the only runtime fast enough.
- **[aqara-g100-detections.md](aqara-g100-detections.md)** — the full catalog of what the G100 detects and how (the design we borrow from).
- **[tasks.md](tasks.md)** — the task breakdown, by phase, v3 first. Each step has a gate.
- **[benchmark.md](benchmark.md)** — how we benchmark against the G100, and against each other.

## The two hardware paths

| | **v3 — Wyze Cam v3** | **v4 — Wyze Cam v4** |
|---|---|---|
| SoC | T31X, single core, MXU2 SIMD (COP2) | T41, MXU3 + **NNA** |
| Runtime | **JZDL on the CPU** (`libjzdl.m.so`) | **venus on the NNA** (`libvenus.so`) |
| Model | our own JZDL `.bin` from the converter (person/pet/package) | yolov5n converted with magik-transform-tools |
| Status | persondet proven (~100 ms, 8.5 % CPU @1 fps, ~2 % motion-gated); **converter cracked bit-exact offline (steps 1–3 done 2026-11-23, [jzdl-format.md](jzdl-format.md)); multi-class head confirmed (any width); QAT training is the open work** | NNA proven (yolov5n ~80 ms, CPU0 only); nmem/L2 fix needed in the image |
| Memory | ~29 MB rmem, tight | ~28 MB free; NNA needs nmem ≥ 16 MB carved |
| We start here | **yes** | after the v3 path is end-to-end |

Same chip family as the G100 (also a T31) on the v3 side, so the G100 is a direct reference for what's achievable.

## The pipeline (both cameras, same shape)

```
ISP → FrameSource ch1 (640×360 NV12) ──┬─→ encoder (sub stream, normal video)
                                       └─→ rvd detection thread (1 frame every N, ~1–2 fps)
                                              │ motion gate (only run the net while IVS reports move + a few s)   ← the cheap trick
                                              │ backend: JZDL (v3) or venus/NNA (v4) → boxes
                                              │ NMS + per-object tracking gate (min track length before report)
                                              ▼
                                       ivs-status {box, class, conf, track_id}  (JSON, class names)
                                              │
                        ┌─────────────────────┼───────────────────────────┐
                        ▼                      ▼                           ▼
                   rmd: per-class         OSD: draw boxes            rhkd: HomeKit
                   filter + region +      on the stream /            Motion Sensor
                   mask + cooldown        snapshot                   + HKSV trigger
                        │
                   split: push (notify / MQTT / send2)  vs  record (HKSV)  — independent per class
```

The two design rules we take from the G100 (aqara-g100-detections.md):

1. **A tracking gate is mandatory.** A raw per-frame detector over-fires; the G100 only reports a track that has survived N frames. This is what makes alerts stable and what lets a static scene cost almost nothing.
2. **Detect, notify and record are three separate decisions, per class.** A class can record to HKSV without notifying, and vice versa.

## Optimising for our hardware

- **Rate, not horsepower.** The G100 trick is 1 fps of 360p, CPU only. We run the detector on the sub stream at ~1 fps, never the video frame rate.
- **Motion-gate the net.** Run the detector only while raptor IVS reports motion (+ a few seconds). On the v3 this took persondet from 8.5 % to ~2 % average, with no missed events on a 10-min test. This is the single biggest win and it's free.
- **v4: pin venus to CPU0** (the NN instructions SIGILL on CPU1) and watch rmem/nmem — the NNA needs a carved region and the L2 128K U-Boot fix.
- **v3: our own small model.** Multi-class via the converter, not a stretched stock model; letterbox the 16:9 frame instead of distorting it (the stock portrait input mislabels cars as people at night).
- **Load only enabled classes** (the G100's `ModelAutoAlloc`): on these tight-RAM parts, don't allocate a model for a class that's switched off.

## Where the work splits

| Piece | Repo / place | Skill to load first |
|---|---|---|
| Detection backend (JZDL multi-class, NNA) | raptor-hal `hal_ivs_*.cpp`, raptor `rvd` | `thingino-dev-persona` |
| Class filter, region/mask/cooldown, split push/record | raptor `rmd` + raptor.conf | `thingino-dev-persona` |
| OSD boxes | raptor OSD / `rvd` | `thingino-dev-persona` |
| HomeKit Motion Sensor + HKSV from classes | rhkd | `thingino-dev-persona` |
| Web UI page (enable/disable, per-class settings) | thingino-firmware webui plugin | `webui-plugin`, `thingino-product-styling` |
| Model converter (v3) | [tools/jzdl/](../tools/jzdl/) | — |
| Benchmark harness | [tools/aitest/](../tools/aitest/) + [benchmark.md](benchmark.md) | — |

See [tasks.md](tasks.md) for the ordered, gated breakdown.
