# Aqara G100: the full detection/event catalog (reference for our own detection handling)

**Why:** a complete map of every detection type the G100 runs, what fires it, what event it emits, and the per-class settings around it — so our own detection handling (raptor `[motion]` classes, rmd, rhkd) is designed from a working commercial example rather than invented. Sourced from strings, exports and the live camera only (firmware-analysis.md, aqara-g100-ai.md). No disassembly, no decryption, nothing copied.

## The detection classes

| # | Class (SDK name) | Input | Method / model | Events it emits | Neural net? |
|---|---|---|---|---|---|
| 1 | **Person** (`figure` / `CObjAlgo`) | 360p NV12 @1–2 fps | object detector, `magik_objdet.bin` (JZDL, shipped) | "person detected" | yes |
| 2 | **Pet** | — | **dead code on the G100** (see below) — no pet model, `ai` never enables it | none in practice | — |
| 3 | **Package** (`CObjAlgo` + package tracker) | same frame | object detector class + appear/disappear tracker | `PACKAGE_APPEARS`, `PACKAGE_DISAPPEARED` | yes |
| 4 | **Vehicle** (`CCarAlgo` + `CCarTracker` + `CCarDB`) | same frame | `car_quality` + `veriwild_r18_4bit` (vehicle re-ID), known-car DB | `Car Near`, `Car Leave`, direction `Up→Down`/`Down→Up`, stranger-car added to DB | yes |
| 5 | **Face** (`CFaceAlgo` + `CFaceTracker` + `CFaceDB`) | person crop | `yunet` (detect) + `face_r18_4bit` (ResNet-18, 4-bit embedder), known-face DB | known member, or stranger (quality + padding gated) | yes |
| 6 | **Pose / gesture** (`CPoseAlgo` + `CDgnnAlgo` + `CPoseTracker`) | person crop | `yolo8_pose_tiny` (skeleton) + graph net (DGNN) on the skeleton | `fun_gesture_event_report` (`GestureResultCallBack`), has a `gesture_mode` and `gesture_recognition` | yes |
| 7 | **Sound** (`CVoiceAlgo`) | PCM, `lumi_pushVoiceData` | audio classifier | `VOICE_CLASS_ID`: baby crying, dog barking, coughing, alarm | yes |
| 8 | **Lens covered / tamper** (`CCoverAlgo`) | 360p frame | classic CV: `cover_threshold`, `NonWhitePercent`, `threshold` | occlusion event | **no** |
| 9 | **Motion** (`CMotionAlgo`; also vidicond IVS) | 360p frame | frame diff + Ingenic IVS "move" | motion heartbeat / region hit | **no** |

**The model list is fixed and small** (from `ai`'s model table, 2026-11-23 disassembly): `yunet`, `face_r18_4bit`, `magik_objdet`, `veriwild_r18_4bit`, `car_quality`, `yolo8_pose_tiny`. **There is no pet model.** All are well-known open *architectures* (YuNet face detector, ResNet-18 4-bit embedders for face and vehicle re-ID, YOLOv8-pose-tiny for skeletons, the Magik object detector) — Aqara's weights are encrypted, but the shapes are reproducible from open models, which matters for "how do we do the same".

**Only `magik_objdet` actually ships, and nothing downloads the rest.** Correcting an earlier claim: the other models are *not* fetched on demand. The only model file on the device is `magik_objdet.bin` (in `encModels/`); `download_file` fetches **known-face / known-car DB photos only, never models**, and nothing in either firmware installs a model. So even when a class is enabled, its model isn't present and it can't run. Confirmed live (our testing, 2026-11-23): enabling face+gesture set the flags (`FACE_1 PERSON_1 MOTION_1` in the frame-rate line) but only `Object Model Load!` appeared — no face/pose/Dgnn model loaded. **Net: on the G100 as shipped, only person detection (objdet, person-only — see below), motion and cover (classic CV) actually run.** Package is enabled in the SDK but unconfirmed (the model emitted nothing non-person when forced on; not tested with parcels). Face, vehicle, pose/gesture and sound are SDK capabilities with no model behind them on this camera.

**Pet is dead code.** The SDK keeps active classes as bits in a word at `UserConfig+0x500` (gesture `0x1`, pet `0x2`, person `0x4`, face `0x8`, package `0x80`, motion `0x100`; car and voice bits not yet mapped). Live default flag on our unit was `0x104` = person + motion. `ai` builds that word in `lumi_get_algo_flag`, which reads face, car, figure, package, gesture, the sound settings, cloud/hap-provisioned and motion — but **never reads `pet_active` and never sets bit `0x2`**. `pet_active` has one reader, the event-report path. So the app's pet toggle never reaches the SDK, there's no pet model to run, and the cat test correctly produced nothing. Pet is advertised in the app but not implemented on this camera.

**Correction on the startup thresholds.** The four values printed at startup (person 0.20, pet 0.70, package 0.50, car 0.60) are just compiled-in *defaults for every possible class*, printed whether or not the class runs. They are **not** evidence that `magik_objdet` is multi-class — the pet line prints even though no pet model exists. So that earlier reasoning was wrong.

**What the object detector actually does — person only, as tested.** Forcing pet (`0x2`) **and** package (`0x80`) on in the SDK (flag 390, `PERSON_1 PET_1 PACKAGE_1 MOTION_1` listed as running) and injecting cat + cow frames produced **no pet and no other non-person event** (people control: 28 events). So `magik_objdet` emits **person only** on this unit; it is not the multi-class model we earlier inferred from the package active bit. Package wasn't tested with actual parcel images, so package is still unconfirmed — but the model gave nothing non-person for animals. If package detection exists at all, it is more likely the package *tracker* over generic/stationary regions than a detector class. See "Open / to confirm" for the package test. (our testing, 2026-11-23)

Classes 4–7 (vehicle, face, pose, sound) are SDK features whose models are **AES-encrypted and licence-bound but not present on this camera and never downloaded** (see above), so they don't run here. 8–9 are plain image/signal processing, no model.

## The event architecture (how a detection becomes an event)

`CCamAlgo::Init` registers **four typed callbacks** (decoded from the mangled signature):

| Callback | Signature | Carries |
|---|---|---|
| Object | `(const tagLumiObjectDetectResult*, OBJ_EVENT, int, void*)` | person / pet / package: the boxes + an `OBJ_EVENT` (detected, `PACKAGE_APPEARS`, `PACKAGE_DISAPPEARED`) |
| Car | `(const tagLumiObjectDetectResult*, int, CAR_EVENT, …, int, int, void*)` | `CAR_EVENT` (near / leave) + direction |
| Voice | `(VOICE_CLASS_ID, int, void*)` | which sound class, + score |
| Generic | `(int, int, void*)` | cover / motion-style (type, value) |

Each then goes out through a per-class reporter: `fun_person_event_report`, `fun_pet_event_report`, `fun_car_event_report`, `fun_package_event_report`, `fun_face_event_report`, `fun_gesture_event_report`. These reach `ha_master` (→ HomeKit Motion Sensor + HKSV trigger) and the cloud agent.

**The common tracking layer gates every object event.** Before a class reports, a tracker runs: assign detections to tracks, fuse duplicates (`Fuse Track, Remove id`), age out stale tracks (`Track %d Out Of Date`), and suppress reports from too-short tracks (`Track %d frame count %d too small, skip report`). Face and car add a "stranger" step on top (`StrangerAddAndReport`: an unknown face/car above a quality gate is added to the DB and reported). So a raw detection never fires an event directly — it must survive tracking first. That is what makes the alerts stable (and what makes the per-frame CPU vary with the scene).

## Per-class settings schema (`persist.app.camera_ai_<cls>_*`)

Not every class has every knob — the shape tells you how each is meant to be used:

| Class | active | sensitivity | region | peroid* | repeat | push | record | trace |
|---|---|---|---|---|---|---|---|---|
| figure (person) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| pet | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| car | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – |
| package | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – |
| face | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | – |
| occlusion (cover) | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | – |
| gesture | ✓ | – | ✓ | – | ✓ | – | – | – (`mode`, `recognition`) |
| voice | ✓ | – | – | – | – | – | – | ✓ |
| sound sub-classes | `alarm_active`, `babycrying_active`, `coughing_active`, `dog_barking_active` | | | | | | | |
| motion | — only `motion_region` | | | | | | | |

\* `peroid` = cooldown between reports (their spelling). Meaning of each:
- **active** — class on/off (note: toggling off in the app doesn't always reach the SDK — on our unit `pet_active=true` but the SDK ran only person+motion; and person kept running with the toggle "off", the privacy finding).
- **sensitivity** — detection/confidence threshold.
- **region** — detection rectangle; **mask** (separate `updateMaskRect`) excludes areas.
- **peroid / repeat** — cooldown, and whether to re-notify while the subject stays.
- **push / record** — notify the app; trigger an HKSV recording. Independent, so "record but don't notify" is possible.
- **trace** — PTZ follow the subject (person, pet, voice, plus `roaming_trace`). G100 is a pan/tilt cam; not relevant to our fixed Wyze cams, but the knob exists.
- debug: `draw_recognition`, `gesture_recognition` overlay boxes; `ptz_*` are pan/tilt calibration, home/sleep points.

## What we take for our detection handling

1. **One reporter interface, many producers.** Each class feeds a common event with {class, boxes, event-type, score}. Map onto raptor `ivs-status` → rmd classes → rhkd. Keep package's appear/disappear and car's near/leave/direction as event *sub-types*, not separate plumbing.
2. **A tracking layer between detection and event is mandatory, not optional.** The min-track-length gate is why the G100's alerts are stable and why a raw per-frame detector would over-fire. Our rmd should gate on a track surviving N frames before it raises a sensor, exactly as `Track … frame count … too small, skip report`.
3. **Per-class knobs, not one global threshold:** active, sensitivity, region, mask, cooldown (`peroid`), repeat, and **push/record split**. This is richer than our current single-threshold plan — put these in the raptor.conf `[motion]` design per class.
4. **Separate "detect" from "notify" from "record".** The G100 keeps push and record independent per class. rhkd should let a class trigger HKSV without a user notification, and vice versa.
5. **Classic-CV classes are cheap wins.** Lens-cover (`NonWhitePercent`/`threshold`) and motion need no model and no NNA — implement them first, on any SoC, as their own sensors.
6. **Sound is a second sensor type.** baby crying / dog barking / coughing / alarm come off the audio path (`lumi_pushVoiceData`), not video. HomeKit has no native sound-class characteristic, but rhkd could expose a second Motion/Occupancy sensor per sound. Needs an open audio model.
7. **Stranger/known DBs** (face, car) are a later feature; note the design (quality+padding gate before adding to a DB) but it's well past our current scope.
8. **Skip for our fixed cams:** `trace`/`ptz_*`/`roaming_trace` — PTZ only.

## Implication for our multi-class model

The G100 validates **person detection only** on a T31 at this model size — `magik_objdet` emitted no non-person class even with pet and package forced on (line 90). It does **not** validate pet (never implemented) nor, so far, package (enabled but no non-person output; untested with parcels). So for our converter target (person/pet/package): **only person has a commercial precedent to lean on; pet and package are both ours to add and prove**, from open data (COCO person; cat/dog for pet; a parcel set for package). Pet and package are therefore the classes that most need a careful accuracy pass — nobody has shown them at this size on this chip.

## Open / to confirm

- Whether **package** is a true `magik_objdet` class or the package *tracker* running over generic object boxes (our testing: an LD_PRELOAD wrapper on `lumi_ActiveAlgo` that ORs in the pet bit `0x2` then reruns the cat frames would also show whether the object model emits any non-person class).
  - **Pet-bit run done (our testing, 2026-11-23 08:01–08:08 GMT, see aqara-g100-ai.md "Pet + package forced on in the SDK"):** with `lumi_ActiveAlgo` wrapped to OR in pet 0x2 + package 0x80, the SDK ran `PERSON_1 PET_1 PACKAGE_1 MOTION_1` (flag 390). Three cat photos and a cow gave **no pet (or any non-person) event**, while the people control gave 28 person events. So `magik_objdet` emits no cat/pet class at the 0.70 pet threshold, i.e. **no non-person class seen**. Package is still untested (no parcel images yet), so "person + package" is still unproven on the package side.
- The `OBJ_EVENT`, `CAR_EVENT`, `VOICE_CLASS_ID` enum integer values (low priority).
- Exact sensitivity→threshold mapping per class (the startup defaults are just constants, not proof of what runs).
