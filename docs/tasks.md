# On-camera AI detection: tasks

Ordered, gated task list for the local-AI detection work. Overview and design: [ai.md](ai.md). Each step has a **gate** — don't start the next until it passes. Items move to a `done.md` (to be created) once verified on a camera. **v3 first, then v4.** Load the skill named in each phase before touching code (see [ai.md](ai.md) "Where the work splits"). Ask the owner before anything runs on a camera.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done. Owner-input gates are marked **ASK**.

## Phase 0 — foundations (mostly in flight)

- [~] **v3 JZDL converter** (../nna/jzdl-format.md, research-3c). **Steps 1–3 done bit-exact offline (2026-11-23):** parser/writer round-trips both samples byte for byte; a NumPy reference (`jzdl_ref.py`) reproduces persondet bit-for-bit (all 11,707 output floats) and an edit test matched the prediction. Arithmetic, weight layouts and type meanings documented. Multi-class head confirmed at any width (so 24 ch for 3 classes is fine). **v3 hardware confirmation done 2026-11-23** (user-approved): real-camera output bit-identical to the NumPy reference and QEMU; camera left clean. QEMU on the build host is now a proven stand-in for step 4. **Remaining: step 4 (QAT + exporter).**
- [~] **Open v3 runtime, ojzdl** ([jzdl-runtime.md](jzdl-runtime.md)). **Steps 1–3 done (2026-11-24):** C99 + MXU2 kernels, bit-identical to libjzdl (36/36 under QEMU, 4/4 on the v3) and as fast on the v3 (persondet forward 84–90 ms vs 89–96 ms). **Next: step 4**, a drop-in for raptor-hal's `hal_ivs_jzdl.cpp`.
- [x] **v4 NNA runtime** proven (yolov5n ~80 ms, ../nna/findings.md). Parked behind the nmem/L2 image change.
- [x] **Benchmark harness** `tools/aitest` exists and measures CPU/RSS/ms on a live camera. See [benchmark.md](benchmark.md).
- [ ] **Decide: detector in `rvd` vs a separate `rad`-style daemon.** The G100 isolates AI in its own process so a model crash can't kill video; our nna.md plan puts it in `rvd`. **Default: in `rvd`** (less IPC, the existing `rvd_jzdl_thread` hook), with a separate process kept as the fallback if a soak test shows instability. Gate: record the decision before writing the backend.

**Gate for Phase 1:** the v3 converter can emit a model the runtime runs with correct boxes — **met: steps 1–3 bit-exact offline AND confirmed bit-identical on the real v3 (2026-11-23)**. Only a trained multi-class model (step 4) is still to come.

## Phase 1 — v3 detection end to end (person only first)

Skill: `thingino-dev-persona`.

- [ ] **Backend: multi-class JZDL in raptor-hal.** Generalise `hal_ivs_jzdl.cpp` decode to `3 anchors × (5 + N)` and N class scores (currently 1 class). Classes = person first, then pet, package.
- [ ] **`rvd` detection thread at ~1 fps on the sub stream**, NV12 in, letterboxed (not stretched) to the model input. Reuse the `rvd_jzdl_thread` path.
- [ ] **NMS + tracking gate.** Port a small tracker: assign → fuse → age-out → **suppress tracks shorter than N frames** before reporting. This is the G100's stability rule.
- [ ] **`ivs-status` carries {box, class, conf, track_id}** as JSON with class names (raptorctl).
- [ ] **Motion gate.** Run the net only while IVS reports move (+ a few s). Verified ~2 % CPU on the v3 in aitest; wire the same gate into `rvd`.
- **Gate:** on the v3, person detection runs at ~1 fps, ≤ a few % CPU motion-gated, correct boxes on a walk-past (use `aitest -s` saved frames to check for misses). **ASK** before the on-camera run.

## Phase 2 — turning detections into actions (v3)

Skill: `thingino-dev-persona`.

- [ ] **`rmd` class filter + per-class settings:** active, sensitivity (threshold), region, mask, cooldown (`peroid`), repeat — mirror the G100 schema (aqara-g100-detections.md). raptor.conf `[motion]`/`[ai]` keys per class.
- [ ] **Split detect / notify / record.** Per class, independent: `push` (notify → MQTT/send2) and `record` (HKSV trigger). A class can do one without the other.
- [ ] **rhkd: HomeKit Motion Sensor per enabled class**, HKSV triggered from the class events (not just raw motion). One sensor, or one per class — decide with the owner.
- **Gate:** turning a class on/off and toggling its push/record changes behaviour on the v3 as configured; a person event triggers HKSV while a pet-only config stays quiet. **ASK**.

## Phase 3 — boxing the objects (v3)

Skill: `thingino-dev-persona`.

- [ ] **OSD boxes on the stream:** draw detection rectangles + class label on the sub (or main) stream via raptor OSD, toggleable (`draw_recognition`-style debug switch).
- [ ] **Boxes in snapshots / HKSV** (optional, heavier) — decide if worth it.
- [ ] **Boxes in the web UI live view** from `ivs-status` (overlay in the browser, no re-encode) — cheapest, do this first.
- **Gate:** boxes line up with real objects on the live view and (if enabled) on the stream.

## Phase 4 — the web UI page

Skills: `webui-plugin`, `thingino-product-styling`.

- [ ] **Plugin manifest + page** (`<name>.webui.json`, `scripts/check-plugins.sh`). Dark Bootstrap + orange `#f80`.
- [ ] **Per-class controls:** enable/disable each class; sensitivity slider; region/mask draw on a snapshot; cooldown; **separate push and record toggles**; live box overlay.
- [ ] **Reads/writes raptor.conf** via raptorctl; no new config store.
- [ ] **Gate on hardware:** the page only offers classes the camera can run (JZDL set on the v3, NNA set on the v4).
- **Gate:** the page sets every raptor.conf key the daemons read, and the live overlay shows current detections. Verified on the v3.

## Phase 5 — v4 (NNA)

Skills: `thingino-dev-persona`; NNA specifics in ../nna/nna.md.

- **Runtime decision (2026-11-23): stay on venus 0.1.2.5 + converter 2.0.4** (the proven pair), but 0.3.7.1 is now a **tested fallback**. Emulator run: our 2.0.4-built `.bin` loads and runs on 0.3.7.1 **byte-identical** to 0.1.2.5 and the v4 — so no reconversion is needed, and the `libdrivers` ORAM rework didn't break our L2-128K/nmem20 layout. We stay on 0.1.2.5 only because 0.3.7.1 gives **no feature gain** at a cost (libvenus 11.2 vs 4.5 MB, +2.2 MB RSS, reverse-engineered shim headers). Reach for 0.3.7.1 (`build.sh ingenic` + shim, no reconversion) only if we later need something it adds. Full analysis + run table: ../nna/venus-0.3.7.1-port.md.
- [ ] **Land the nmem/L2 image change** (U-Boot `CONFIG_T41_L2_128K`, branch `t41-nna-oram`; ../nna/todo.md). **ASK** — needs a flash and an SD card for the payload.
- [ ] **venus backend behind the same `ivs-status` interface** as the JZDL one, pinned to CPU0. Same tracker, same rmd/rhkd/OSD/webui above it — only the backend differs.
- [ ] **Richer models** (more COCO classes) once the base path works; the NNA has the headroom the T31 doesn't.
- **Gate:** the v4 produces the same `ivs-status` shape as the v3, so Phases 2–4 work unchanged on it.

## Phase 6 — benchmark + upstream

- [ ] **Benchmark against the G100** per [benchmark.md](benchmark.md): CPU, RSS, latency, and a detection-quality pass on the same frames. Record v3-JZDL vs v4-NNA vs G100.
- [ ] **PR resource stats** on every raptor/rhkd change (`tools/rhkd-stats.sh` rule): CPU/mem/flash tables.
- [ ] **Upstream the generic pieces** (see ../upstream/upstream.md): rmd multi-class filter, the webui AI page, the raptor-hal JZDL multi-class decode and the NNA backend. Keep household/LAN specifics out.

## Phase 7 — sound detection (audio, parallel track)

Skill: `thingino-dev-persona`. A different modality from the video classes (audio, not frames), so it can proceed in parallel once Phase 2's event plumbing exists. The G100 does this with `CVoiceAlgo` on the mic PCM (../nna/aqara-g100-detections.md).

- [ ] **Audio classifier model** — small CNN on log-mel features, open data (AudioSet / ESC-50 subsets). Classes: baby crying, dog barking, glass break, smoke/CO alarm. Audio models are tiny, so plain CPU; export via `jzdl`/`magikconv` only if needed.
- [ ] **Feed from the `rad` audio path** (PCM), like the G100's `lumi_pushVoiceData`. Low rate, CPU-cheap, gated on sound energy.
- [ ] **Same event interface** → rmd/rhkd. HomeKit has no sound-class characteristic, so expose each sound as its own sensor (Occupancy/Motion or a programmable switch) and/or MQTT.
- [ ] **Web UI** — per-sound enable + sensitivity, alongside the video classes.
- **Gate:** a clap / alarm test triggers the right sensor; a quiet room stays silent. **ASK** before the on-camera run.

## Tools we build (inventory)

| Tool | Status | Where |
|---|---|---|
| `aitest` — on-camera CPU/RSS/latency harness | exists | ../nna/tools/aitest/ |
| `jzdl/` — JZDL parser/writer/converter (v3 models) | in progress | ../nna/tools/jzdl/ |
| QAT training pipeline (person/pet/package, open data) | not started | to add under `ai/train/` |
| Benchmark runner + result log | partly (aitest) | [benchmark.md](benchmark.md) |
| webui AI plugin | not started | thingino-firmware |

## Open decisions (ASK the owner)

- Detector in `rvd` or a separate daemon (Phase 0) — default in `rvd`.
- One HomeKit Motion Sensor for "any class", or one sensor per class (Phase 2).
- Whether to draw boxes on the encoded stream/HKSV, or only in the web UI (Phase 3).
- v4 flash + SD card for the NNA payload (Phase 5).
