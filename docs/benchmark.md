# Benchmarking our AI detection against the Aqara G100

The G100 is a commercial local-AI camera on the **same SoC family as our v3** (T31), so it's the yardstick: if our v3 can't match it, the design is wrong. This doc is how we measure, so numbers from different runs and cameras are comparable. Live G100 numbers and the method that produced them are in ../nna/aqara-g100-ai.md; the detection design we compare against is in ../nna/aqara-g100-detections.md.

## What we compare

| Axis | Metric | Why |
|---|---|---|
| **Cost** | CPU % of one core, RSS, flash, rmem/nmem | these parts are tight; cost decides if it ships |
| **Latency** | ms per frame (grab / pre / infer / post), end-to-end detect→event | responsiveness of an alert |
| **Rate** | frames/s actually run | the G100's trick is low fps, not horsepower |
| **Quality** | detections on a fixed frame set: hit / miss / false, per class | a cheap detector that misses people is useless |
| **Stability under gating** | CPU on a static scene vs a busy one; events on a known walk-past | the tracking + motion gate is where the budget is won |

## The three things we put side by side

1. **G100** — Aqara's shipping stack (T31, JZDL CPU, retrained persondet + package). Measured read-only on the live camera, or via the LD_PRELOAD frame shim for a controlled frame set (../nna/aqara-g100-ai.md). We never decrypt or reuse its models.
2. **v3 (ours)** — JZDL on the T31 CPU, our own multi-class model.
3. **v4 (ours)** — venus on the T41 NNA.

## How to measure (the harness)

`tools/aitest` (../nna/tools/aitest/) is the common harness. A separate process asks `rvd` for one 640×360 NV12 sub frame (`save raw`, no raptor.conf change), runs a backend, and logs per-phase ms, its own CPU %, `rvd` CPU %, system busy %, RSS and free memory. It mirrors the G100's own design (one sub frame per second, CPU only), so the comparison is like-for-like.

Keep these fixed across runs so numbers compare:

- **Rate:** 1 fps unless stated. Report the measured fps, not the target.
- **Input:** 640×360 NV12 from `rvd` ch1. Letterbox to the model input; note if stretched.
- **Colour:** note BGR vs RGB (`AITEST_BGR=1`); it shifts scores slightly.
- **Gating:** report ungated and motion-gated (`-g <seconds>`, polls `ivs-status` like rmd does) separately. The gated number is the one that ships.
- **Duration:** ≥ 3 min for CPU; ≥ 10 min for a gated event count.
- **Scene:** say whether static or busy, day or IR — the G100 proved CPU is scene-dependent (4.5 % night static → 13–22 % daytime).

## The fixed frame set (quality)

For quality, measure every target on the **same frames**, not live scenes:

- **Our side (v3/v4):** feed the frames through `aitest` (NV12 files) and record hit/miss/false + score per class.
- **G100 side:** the LD_PRELOAD frame shim feeds the same frames to its running `ai`, read the event reports. (Live-scene / tablet is a fallback; screens add glare, moiré and IR issues — daylight only, compare hit/miss not scores.)
- Frame set: 30–50 images across person, pet (cat/dog), package, plus a vehicle and an empty control, each also as a 16-px-shifted pair for a motion variant. Keep them in `ai/data/` (to create) so runs are repeatable.

## The results table (fill one per milestone)

| Run | Target | Model | Input | fps | Infer ms | CPU % (ungated / gated) | RSS | rmem/nmem | Quality (P/Pet/Pkg) | Date |
|---|---|---|---|---|---|---|---|---|---|---|
| ref | G100 | objdet (person-only) | 640×360 | 1 | ~40 | 4.5 (static) / 13–22 (day) | 3.8 MB | none | — | 2026-11-22 |
| — | v3 | persondet (1 cls) | 416×320 | 1 | ~102 | 8.5 / ~2 | 3.6 MB | — | person only | 2026-11-23 |
| — | v3 | ours (person/pet/pkg) | tbd | | | | | | | |
| — | v4 | yolov5n (NNA) | 320×192 | | ~80 | (NNA, CPU0) | | 13 MiB nmem | | 2026-11-19 |

Baselines already measured are in ../nna/findings.md ("CPU detection, G100-style"). Each new row cites the aitest log.

## What "winning" looks like

- **v3 matches the G100's budget:** person detection at ~1 fps, motion-gated, within a few % CPU on the single T31 core, RSS in single-digit MB, no false alarms on street traffic/trees/light.
- **Multi-class (pet/package) added at roughly the same cost** (same network, wider head) — the class we have no commercial precedent for, so it needs the quality pass most.
- **v4 does more** (more classes, higher input) at NNA cost the T31 can't reach, so the two cameras land in different places by design — record both, don't force one number.

## PR reporting

Every raptor/raptor-hal/rhkd change that touches this carries CPU/mem/flash tables (the `tools/rhkd-stats.sh` rule). The benchmark rows above are the before/after for those tables.
