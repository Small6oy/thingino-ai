# On-camera AI detection: tasks

Ordered, gated task list for the local-AI detection work. Overview and design: [ai.md](ai.md). Each step has a **gate** — don't start the next until it passes. Items move to [done.md](done.md) once verified on a camera (first move 2026-11-26). **v3 first, then v4.** Load the skill named in each phase before touching code (see [ai.md](ai.md) "Where the work splits"). Ask the owner before anything runs on a camera.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done. Owner-input gates are marked **ASK**.

## Phase 0 — foundations (mostly in flight)

- [~] **v3 JZDL converter** ([jzdl-format.md](jzdl-format.md)). **Steps 1–3 done bit-exact offline (2026-11-23):** parser/writer round-trips both samples byte for byte; a NumPy reference (`jzdl_ref.py`) reproduces persondet bit-for-bit (all 11,707 output floats) and an edit test matched the prediction. Arithmetic, weight layouts and type meanings documented. Multi-class head confirmed at any width (so 24 ch for 3 classes is fine). **v3 hardware confirmation done 2026-11-23** (user-approved): real-camera output bit-identical to the NumPy reference and QEMU; camera left clean. QEMU on the build host is now a proven stand-in for step 4. **Remaining: step 4 (QAT + exporter).**
- [ ] **Decide: detector in `rvd` vs a separate `rad`-style daemon.** The G100 isolates AI in its own process so a model crash can't kill video; our NNA plan puts it in `rvd`. **Default: in `rvd`** (less IPC, the existing `rvd_jzdl_thread` hook), with a separate process kept as the fallback if a soak test shows instability. Gate: record the decision before writing the backend. **2026-11-26: in `rvd` in practice:** both images since 2026-11-25 run detection inside rvd (v3 ojzdl, v4 onna); not formally confirmed by the owner, so left open.

**Gate for Phase 1:** the v3 converter can emit a model the runtime runs with correct boxes — **met: steps 1–3 bit-exact offline AND confirmed bit-identical on the real v3 (2026-11-23)**. Only a trained multi-class model (step 4) is still to come.

## Data v2 — more data + vehicle class (owner, 2026-11-24, in progress)

Classes become **person, pet (cat + dog), package, vehicle (car, truck, bus, van, motorcycle; bicycles stay background)**, so heads go 3 × (5 + 4) = 27 ch. Same graph and cost on both cameras.
- [ ] **Label hygiene:** every image is labelled for all 4 classes where its source has them. Today 5,097 of the 32,178 COCO train images (16 %) hold unlabelled cars/trucks, so the model learns vehicles as background.
- [ ] **COCO:** car (12,251 imgs), truck 6,127, bus 3,952, motorcycle 3,502. Keep small (< 5 % of frame) pets: 2,928 instances.
- [ ] **Open Images:** Car 89k imgs, Truck 8k, Bus 7k, Van 5k, Motorcycle 7k; Cat 12.5k + Dog 20.6k imgs, which add many outdoor/distant pets. OI "Box" has only 2,212 images in total (close-ups), so OI can't fix package.
- [ ] **Roboflow Universe (CC BY 4.0 only; verify each licence on its page, record attribution in the manifest):** parcels "parcel-detect" (~3.1k imgs), pacchi "parcel" (730), zouhj "parcel" (301); CCTV pets "CatoCam" (top-down, IR night), "Dogs_add_camera". Needs the owner's Roboflow API key (keep it out of every repo). Excluded: quailrobot parcels (NC), Intruder Detection (DbCL).
- [ ] **Manifest v2** in `train/manifest/` + `data_prep.py` with 4 classes; `NC`/class names read from the manifest instead of the hardcoded 3; `init_persondet` widens 1 → 4 classes; build_model picks it up. Via PR.
- [ ] **Roboflow candidates for later (found 2026-11-24, licences not yet checked; fetch_roboflow.py checks live):**
  - security cameras, person + vehicle: chanyobjectdetection/security-cam-vision (~2k), remotesurveillance/remotesurveillancenight (~4.6k, person/car/bicycle at night), bagas-dwi-s/vehicle_person_night
  - doorbell view: front-door-camera/ring-door-camera (car, dog, mail truck, person, stroller), craig-test-workspace/doorbell-camera-alert, smartdoorbell/cats, doorbellframe2/newdog + cats
  - **hard negatives / future "animal" class** (raccoon, fox, deer, rabbit seen as false "pet"): workspace-mxy0x/ipcams2 (home cams, day + IR), raccoon-xysmf/animal-detection (raccoon, badger, fox; private cams)
  - dogs from cameras: dogsaddcamera/dogs_add_camera_* (several have no exportable version)
  - NOT thermal (FLIR, calib-buckton/infrared, roboflow-100/flir-camera-objects): our cameras are near-IR greyscale, where people/animals look nothing like thermal "hot = bright"
- [ ] **Own camera frames: NOT YET (owner, 2026-11-24).** Later: low-rate private frame grabs (v3 + v4, day + IR) for training, T41 calibration, and a ~200-frame real-camera test set.
- **Gate:** a v2 build beats v1 on person by no more than noise, and clearly on pet/package (on more than one run), with vehicle AP reported. Re-test live on the v3 (box on car, pets).

## Pipeline synergy — exporter, model library, ojzdl + onna (owner, 2026-11-25)

Going forward with our own runtimes (ojzdl on the T31, onna on the T41). Exporter, library and runtimes must agree by construction. Order as listed. Split with the onna session: we take the exporter/library items first unless agreed otherwise.
- [ ] onna CPU leftovers (-T on the v4): unfolded pc ops (YOLO-N 11.5 ms: the 0.5x after residual means), concat rescales (10-15 ms), nearest-upsample deconv (4-8 ms), adds 5 ms.
- [ ] Anchors in the .bin? (not found yet; raptor-hal falls back to YOLOv5's unless a .qgraph gives them). *(Review 2026-11-26: model.conf now carries anchors (#57/#58) and onna takes them from the caller (gtxaspec/raptor-hal#25); possibly moot. Owner/onna session to confirm.)*
- [ ] **1. Exporter emits everything onna needs:** `export_t41.py` also runs `mkgraphq.py` → `model.qgraph`; the library's T41 folders carry it (v1 and v2 have none today, so onna can't load a library model as is). *(Review 2026-11-26: likely obsolete: onnaq builds its graph from the .bin since #54 and the .qgraph is optional with model.conf. Confirm before deleting.)*
- [ ] **3. Op coverage matrix + export-time refusal:** one table of the layer kinds the exporter can emit vs what ojzdl and onna run; the exporter refuses a graph a target runtime can't run. Today: persondet graph = ojzdl yes, onna chain only (needs Add); YOLO-N = ojzdl yes, onna needs Add + nearest upsample + stride-2 dw + 3 heads.
- [ ] **5. One preprocessing definition:** the card's `preprocess` is the source of truth (today: 16:9 frame stretched to the input; the T31 v1 card wrongly says centre crop); letterbox becomes a per-model option. Do with the letterbox PR.
- [ ] **Optional, larger: own T41 converter.** Write the T41 `.bin` ourselves (format largely decoded for onna: W headers, B/T requant) instead of Ingenic's magik-transform-tools. Removes the Clip/Add workaround and the closed tool; exact T41 export as for the T31.

## Open pretrained starts — publishable models (owner, 2026-11-25)

Goal: models trained from openly licensed weights (no Ingenic weights → publishable). **Merged: PR #41** (c5be435); doc [pretrained.md](pretrained.md). Weights sha256-pinned, licences saved. **No training on our machines (owner): runs go to the teams with GPUs.**
- [ ] Optional for mnv4n speed on the T31: replace its two mid-graph dense convs with dw s2 + 1x1 (81.5M → ~30M MACs on the slow path).
- [ ] **Hand to a GPU team:** `build_model.py --arch yoloxn` and `--arch mnv4n` (full runs), then compare with v2 persondet (.463) and YOLO-N (.490) on the full val; v3 speed + live.
  - **yoloxn 4-class back 2026-11-25** (release model-full-yolox-yoloxn-v1): full val T31 exact person .407 pet .616 package .643 vehicle .340, **mean .501** (> YOLO-N .490, > persondet .463) but **person lowest of the three** (small person .075). T41 int8 sim .507 (holds). Gate PASS. Live v3: 132 vs 92.5 ms/s, sees the parked car; reverted. **Owner 18:52: default on v3 + v4, library + upstream.** v3 default since 18:53 (/usr/share/models/yoloxn_v1_t31.bin = library re-export, raptor.conf backup kept on the camera). Model library PR merged (T31 status default, T41 experimental; conform PASS both). Public release [model-yoloxn-v1](https://github.com/Small6oy/thingino-ai/releases/tag/model-yoloxn-v1) (Apache-2.0). **v4 default since 19:22**: onna in rvd 75 ms network, no reset in 15 min. [x] v4 image so yoloxn survives a reboot: **image 20261125-2114, flashed 2026-11-26 05:33, runs yoloxn from the image, no binds.** onna wcache[256] overflow (yoloxn 324 records) fixed: #76 -> gtxaspec/raptor-hal#25 -> our HAL integration branch (T41 ONNA=1 + T31 OJZDL=1 build clean). The fixed rvd went on the v4 (owner OK); then daylight check. mnv4n and the 3-class / 384x224 runs still to come.
  - **2026-11-26: GPU full runs back** (yolon, scratch-yoloxn, mnv4n), scored against yoloxn v1. **Scored 2026-11-26 09:05, full val T31 exact mean4 / person:** scratch-yoloxn .491 / .391, full-yolon .473 / .463 (persondet init, private), mnv4n .419 / .332; none beats yoloxn v1 (.501) or the YOLO-N helper (.490, person .477). All gate PASS; T41 int8 holds; onnaq byte-identical for yolon + yoloxn, **mnv4n fails in onnaq ("more Add ops than add records (124)")**. No camera trial yet (owner to decide).
  - **Owner 2026-11-25: package dropped for now → 3 classes person, pet, vehicle.** `build_model.py --classes LIST` (branch `classes-3`, now on main) (label ids remapped, other classes' boxes dropped), hand-off [scripts/train-open-3class.sh](../scripts/train-open-3class.sh) `[yoloxn|mnv4n|both]` (quick, then full).
  - [x] **Untrained scores, full val 3 classes** (`train/eval_start.py`): YOLOX-Nano as released 0.657 / 0.662 / 0.556 (mean **0.625**, above trained YOLO-N .465 / persondet v2 .421 on these 3); with the T31 changes **0** (ReLU alone kills it), our graph float/BN/int8 ~0. As-is on the cameras = timing only.
  - [x] **Live speed, untrained 3-class yoloxn:** v3 rvd ojzdl **131 ms**/inference vs v2 persondet 93; v4 onnaq -e nna **139 ms** avg (pessimistic: s2 dw on CPU), heads byte-identical to host.
  - [ ] Open YOLO-N weights: only Ultralytics yolov5n (AGPL-3.0). Recommendation: don't; yoloxn is the Apache-2.0 equivalent (same CSP/PAFPN/3-head family).
  - [x] **v4-only YOLOX-Nano as released, no training (owner said yes, 2026-11-25): doesn't work via magik PTQ.** Script `t41yolox.py` (SiLU, SPP, own anchor-free head; Focus+BGR folded into an exact 6x6 stem, diff 5.6e-5; 7 COCO rows → person/pet/vehicle).
    - magik 2.0.4 converts SiLU (Sigmoid×Mul) + stride-1 MaxPool fine (1.08 MB), but only when every weight is an ONNX initializer: Constant-node weights (tensors made at trace time) convert "ok" yet give garbage (cosine −0.5); torch 2.9 needs `dynamo=False`; F.pad / Gather / Resize crash magik.
    - Float twin 0.667 mean on 600 val frames (0.694/0.702/0.605). **int8 (mgkrun) 0.05–0.07 mean on 40 frames**; per-channel corr 0.13–0.75, head cosine 0.89–0.95. KL default best; MSE worse (0.0); MAX_ABS not accepted (KL/ADMM/MSE only); ADMM too slow, stopped. Split box/obj/cls outputs: no change → loss is in the trunk.
    - Zero-shot activation swap no help: hardswish 0.146 vs SiLU 0.616 float (300 frames).
    - Conclusion: released weights don't survive PTQ; a v4 YOLOX needs quantisation-aware fine-tuning (GPU team), same as the T31 path. Not tried: magik BIAS_CORRECT / bigger calibration set.
  - **2026-11-25 afternoon (owner: GPU run in progress, optimise v3 speed):**
    - [x] raptor-hal vendored copies were already current: ojzdl has PR #43 in gtxaspec/raptor-hal#22 / #25 / #26; onna synced to #52 (c7a8af5).
    - [x] model.conf for yoloxn 3-class comes out of the build automatically (classes, anchors_8/16/32, input).
    - [x] **ojzdl speed, PR #65:** MXU2 eltwise (type 78: 10.0 → 1.7 ms) + head kernel (8.4 → 6.6 ms); yoloxn v3 layer sum 107 → ~95 ms, persondet unchanged. Gate 153/153.
    - [x] **Regression fixed, PR #63:** #61 made ojzdl refuse every graphgen model (yoloxn, mnv4n, the model in training) — upsample P[1] check; mode 1 doesn't use P[1]. graphgen now writes the real bits.
    - [x] **Input size, PR #66 `--input WxH`:** YOLOX-Nano as released 384x224 0.639 vs 416x320 0.627 (448x256 0.652); **v3: 384x224 = 75 ms standalone / 85 ms in rvd** (416x320: 126 / 131; persondet v2 93 in rvd). rvd scales 640x360 → 384x224 itself. Hand-off: `INPUT=384x224 scripts/train-open-3class.sh yoloxn`.
    - [ ] Next GPU run: yoloxn at 384x224 (owner to ask the team when the 416x320 run is done, or in parallel).
    - [x] raptor-hal `src/ojzdl` synced 2026-11-25 16:20: gtxaspec/raptor-hal#22 at 8791085 (#63 + #65 on top of #61/#62), merged into our HAL integration branch. Builds clean: T31 OJZDL=1, T41 ONNA=1.
    - [x] **Frame input 1.5x faster on the T41 (PR #70, 2026-11-25):** `ojzdl_resize_nv12` replaces the HAL's per-pixel NV12 -> RGB (9.5 of 19.3 ms on the v4): same bytes (nv12test 680/680 on the v4, resize gate vs libjzdl 40/40, gate.sh regression 54/54), v4 19.3 -> 12.8 ms (x86 5.1 -> 1.9). raptor-hal: synced into gtxaspec/raptor-hal#22 (+ T31 backend switch) and #25 (+ switch), both merged into our HAL integration branch (keeps the per-class thresholds). **Step 2 (PR #71, e7a5905):** vertical pass on MXU3 (`-DOJZDL_MXU3`, core 0 only), same bytes (680/680 host == T31 MXU2 == T41 scalar == T41 MXU3, under QEMU `-cpu XBurst2-MXU3.1` and on the v4; resize gate 40/40), v4 12.8 -> **10.8 ms** (rvd running, no reset). raptor-hal #22 / #25 (ONNA_FLAGS += -DOJZDL_MXU3) updated. **Step 3 DONE (PR #73, c1b84b1, 19:50):** conversion + horizontal on MXU3 too (planar), v4 640x360 -> 416x320 **3.16 ms** (448x256 3.07, 384x224 2.67; horizontal alone 5.58), faster than raptor-hal#21's 5.7; same bytes (680/680 on 4 builds and on the v4; gate 40/40; models 54/54). raptor-hal #22 `6da129d`, #25 `5a7be85`. Not yet in a camera image / rvd (9.5 ms on the T41, which does not run MXU2) toward #21's 5.7 ms.
    - [x] Leftover remote branch `ojzdl-speed` on GitHub (superseded by #65, PR #64 closed); delete when convenient. **Gone from GitHub (checked 2026-11-26).**
- [ ] Watch in QAT: a learned scale could re-cross the requant limit mid-training (export each epoch would fail); fit_requant only runs at the QAT start.
- [ ] Stride-2 depthwise pads: test whether libjzdl/ojzdl honour top/left pads (`1 0 1 0`) → YOLOX/MNV4 keep their own padding.
- [ ] T41: export of these graphs (export_t41 build_onnx_graph) + onna layer kinds (Add, nearest upsample, stride-2 dw, 3 heads: the synergy list). A T41-only training path would let MNV4 / YOLOX-Tiny / NanoDet keep signed activations. *(Review 2026-11-26: the onna layer kinds are done (#45/#47/#50/#52) and yoloxn/mnv4n run on the v4; whether export_t41 itself is finished is not clear from git.)*

## Model options — 4 starting points, each for T31 + T41 (owner, 2026-11-24)

`build_model.py` grows `--arch persondet|yolon` × `--init scratch|pretrained` (+ the open pretrained inits above):
1. **persondet, scratch:** PR #15 (`--init scratch`). Publishable. Short learning check (8k imgs, 10 float epochs): float mean 0.112, after BN-fold + calibration 0.096, so it survives quantisation.
2. **persondet, Ingenic:** today's default. Private.
3. ~~TinyDet~~ **REMOVED (owner, 2026-11-25):** too slow on the T31 (317–834 ms vs persondet ~104), weights' terms unconfirmed. Port deleted (PR #40), model library t41-mars entry deleted. Its format findings stay in [jzdl-format.md](jzdl-format.md) (type 46 rules).
4. **YOLO-N, Ingenic JZDL yolov5 sample init** (owner chose; private): `magik_model_yolov5.bin` = 112 layers, input 320x416 (portrait), 3 heads 27 ch (4 classes), ~200 ms on the v3. New vs persondet: **type 78 = element-wise add** (residuals, params [bits, 20]), **type 69 upsample param[4]=1** (persondet uses 0 = zero-insert; 1 is probably nearest), dw convs with stride 2, 3 heads (strides 8/16/32).
   - [ ] jzdl_ref support for types 78 and 69/1, then bit-exact vs QEMU on the sample (the same ladder as persondet steps 1–3).
   - [ ] net.py YOLO-N graph with exact integer arithmetic, then init from the sample, then QAT; export via skeleton; T41 via ONNX.
- Order: YOLO-N first (it fits the T31 for sure; ~2x cost for accuracy).

## Phase 1 — v3 detection end to end (person only first)

Skill: `thingino-dev-persona`.

**2026-11-24 evening: first live run inside raptor DONE:** raptor-hal shared decoder (now gtxaspec/raptor-hal#26: reads the model's head layout from its output trailer; per-class thresholds; anchors per stride; matches a Python reference on 12/12 outputs and fixes the yolov5 sample's near-zero scores) + a raptor rvd branch (`class_thresholds`, `class_names` -> `label` in ivs-detections, `anchors_8/16/32`, `detect_interval_ms`). On the v3 at 1 Hz: +7.7 % CPU, +3.9 MB RSS, video untouched; person 0.45–0.50 on a walk-past; package missed, plus a frame-sized false "package" at 0.20–0.22. Branches local at the time (upstream PRs need the owner's OK; the HAL branch sits on #22).
- [ ] Next: max-box-area filter (or package threshold 0.25+), motion gate, tracking gate (N frames), then upstream PRs. *(Review 2026-11-26: tracking gate done 2026-11-25 (see done.md); box-area filter, motion gate and upstream PRs still open.)*
- 2026-11-24 21:15: helper **YOLO-N 4-class** (epoch 26) scored + tested live: full val (6,223) mean 0.490 (person .477, pet .558, package .564, vehicle .360; small objects weak), T41 int8 cosine >= 0.994, live v3 +15 % CPU, found the far cars only (no person logged, parcel missed). **Tomorrow (owner):** compare with the other helper builds (v2 persondet, scratch) on the same full val and live; not yet in the model library.
- 2026-11-25: **v2 persondet GPU run** arrived. 30 epochs, bs 64, lr 1e-3, 69,854 train / 3,000 val, Ingenic persondet init widened to 4 classes. Best = epoch 23: own-val AP50 person .452, pet .522, package .655, vehicle .266, mean .474 (v1 was .341). **Full val (6,223) results 2026-11-25:** person .440, pet .556, package .592, vehicle .266, mean .463 (YOLO-N .490: v2 wins package, ties pet, loses person -.04 / vehicle -.09, at half the CPU). T41 int8 (400): .454/.637/.554/.282, cosine .995. Live v3 in rvd: +7 % CPU, only the parked car (vehicle .41-.60) at dawn. **In the model library:** `t31-jzdl/...416x320/v2` + `t41-venus/...416x320/v2`, candidate; public catalog PR #39 (TinyDet entry kept out of the public catalog). - [x] v2 on the v4 NNA (2026-11-25, clip=chain re-export of ep23, same graph): venus 34.5 ms avg / 23.1 best, onna 40.7 / 25.1, heads byte-identical venus = onna = golden416.py; chain int8 worse than the library's Add-clamp file (pet .540 vs .637), which was not yet on silicon then. - [x] onna Add-clamp support, then run the library T41 v2 file on the v4: **done 2026-11-25** (#45/#52; library v2 file byte-identical to the PC executor on the v4, 26.3 ms avg). - [ ] live person/pet/package walk-past on the v3.

**NEXT UP (owner, 2026-11-24): run the trained model live inside raptor.** Where we are:
- The model exists: model library `t31-jzdl/detect-person-pet-package-416x320/v1` (GPU run, mean AP50 0.341). It is bit-identical on the real v3. Live under `aitest` (not raptor) it runs at ~104 ms and 8.4 % CPU at 1 fps, the same as persondet. Person works live; pet/package scores are too low at 0.40.
- **Contract mismatches to settle before wiring (found 2026-11-24):**
  - [x] (done 2026-11-24, shared decoder + rvd branch) **Anchors.** Upstream `hal_ivs_jzdl.cpp` decodes with the *stock YOLOv5* anchors, but our model keeps **persondet's** two heads and anchors (stride 32: 81,82 135,169 344,319; stride 16: 10,14 23,27 37,58) plus persondet's output layout (heads + shape trailer, which `aitest` `headed_decode` reads). Either add an `anchors=` key / persondet-layout decode to raptor-hal (small, generic) or retrain with stock anchors. Preferred: the raptor-hal option, so persondet-derived models work as they are.
  - [ ] **Letterbox vs stretch.** Our model is trained on the 16:9 frame **stretched** to 416x320 (as raptor feeds persondet today). The planned letterbox PR (below) would change the input it sees. Make letterbox a per-model option (`letterbox=0|1`), default off for our model, or retrain letterboxed.
  - [x] (done 2026-11-24, shared decoder + rvd branch) **Per-class thresholds.** `conf_threshold` is one value today; add `conf_threshold_<class>` (or a list), e.g. person 0.40, pet/package 0.20. Interim until better pet/package data lands.

- [ ] **Match the upstream model contract:** our exported JZDL `.bin` uses a standard YOLOv5 head with the stock anchors and ships a names `.txt` (person, pet, package). Check: `num_classes=3` loads it with no code change. *(Review 2026-11-26: likely superseded by the shared YOLO decode (gtxaspec/raptor-hal#26: head layout from the model, anchors and class names from model.conf).)*
- [ ] **Letterbox PR for raptor-hal** (upstream squeezes the 16:9 frame into the model input — the night car-as-person problem). Small, generic, upstreamable on its own.
- [ ] **`rvd` detection thread at ~1 fps on the sub stream**, NV12 in, letterboxed (not stretched) to the model input. Reuse the `rvd_jzdl_thread` path. *(Review 2026-11-26: running in rvd on both cameras (detect_interval_ms, gtxaspec/raptor#96); still stretched, not letterboxed: letterbox is the open part (#26 preprocess field).)*
- [ ] **`ivs-status` carries {box, class, conf, track_id}** as JSON with class names (raptorctl).
- [ ] **Motion gate.** Run the net only while IVS reports move (+ a few s). Verified ~2 % CPU on the v3 in aitest; wire the same gate into `rvd`.
- **Gate:** on the v3, person detection runs at ~1 fps, ≤ a few % CPU motion-gated, correct boxes on a walk-past (use `aitest -s` saved frames to check for misses). **ASK** before the on-camera run.

## Phase 2 — turning detections into actions (v3)

Skill: `thingino-dev-persona`.

- [~] **Class filter, first step done 2026-11-25 in rvd, not rmd** (rhkd reads rvd's raw motion, so a filter in rmd would miss HomeKit): raptor rvd branch, `[motion] classes = person,pet,...`; ivs-status `class_mask` for the per-class actions below. Tested on the v4 (parked car no longer fires the HomeKit sensor, a person does). **Tracking gate done 2026-11-25 (`[motion] still_sec`, default 10 s):** G100 design rule 1; `classes` now = Detect per class (owner: detection and motion are different things), motion = a detected object that is new or moving; parked car tested on the v4. Note: every rvd restart makes the objects in view "new" once. Still open: **`rmd` per-class settings:** active, sensitivity (threshold), region, mask, cooldown (`peroid`), repeat — mirror the G100 schema ([aqara-g100-detections.md](aqara-g100-detections.md)). raptor.conf `[motion]`/`[ai]` keys per class.
- [~] **2026-11-25: one sensor per class (owner), written:** rhkd branch with Person/Pet/Package/Vehicle sensors + main (HKSV) sensor on record_classes; builds clean, not on a camera yet. Was: **rhkd: HomeKit Motion Sensor per enabled class**, HKSV triggered from the class events (not just raw motion). One sensor, or one per class — decide with the owner.
- **Gate:** turning a class on/off and toggling its push/record changes behaviour on the v3 as configured; a person event triggers HKSV while a pet-only config stays quiet. **ASK**.

## Phase 3 — boxing the objects (v3)

Skill: `thingino-dev-persona`.

- [ ] **OSD boxes on the stream:** draw detection rectangles + class label on the sub (or main) stream via raptor OSD, toggleable (`draw_recognition`-style debug switch).
- [ ] **Boxes in snapshots / HKSV** (optional, heavier) — decide if worth it.
- [ ] **Never enable `IVS_DETECT` with the JZDL backend on the v4:** JZDL's X2000 build uses MSA, the T41 has none, `rvd` dies with SIGILL the moment `algorithm=yolo` is chosen and video stays down until it's changed back.
- **Gate:** boxes line up with real objects on the live view and (if enabled) on the stream.

## Phase 4 — the web UI page

Skills: `webui-plugin`, `thingino-product-styling`.

- [ ] **Per-class controls on top of it:** enable/disable each class; sensitivity slider; region/mask draw on a snapshot; cooldown; **separate push and record toggles**; live box overlay.
- [ ] **Reads/writes raptor.conf** via raptorctl; no new config store.
- [ ] **Gate on hardware:** the page only offers classes the camera can run (JZDL set on the v3, NNA set on the v4).
- **Gate:** the page sets every raptor.conf key the daemons read, and the live overlay shows current detections. Verified on the v3.

## Phase 5 — v4 (NNA)

Skills: `thingino-dev-persona`; NNA specifics in [nna-runtime.md](nna-runtime.md).

- **Runtime decision (2026-11-23): stay on venus 0.1.2.5 + converter 2.0.4** (the proven pair), but 0.3.7.1 is now a **tested fallback**. Emulator run: our 2.0.4-built `.bin` loads and runs on 0.3.7.1 **byte-identical** to 0.1.2.5 and the v4 — so no reconversion is needed, and the `libdrivers` ORAM rework didn't break our L2-128K/nmem20 layout. We stay on 0.1.2.5 only because 0.3.7.1 gives **no feature gain** at a cost (libvenus 11.2 vs 4.5 MB, +2.2 MB RSS, reverse-engineered shim headers). Reach for 0.3.7.1 (`build.sh ingenic` + shim, no reconversion) only if we later need something it adds.
- [ ] **Test [raptor-hal#21](https://github.com/gtxaspec/raptor-hal/pull/21) (andylee77, Uranus/Magik 3.x) on the v4** instead of writing a parallel venus backend. It already sits behind the `hal_jzdl_*` API (rvd unchanged), pins runtime calls to core 0, shares the YOLOv5 decode with JZDL. Needs [raptor#97](https://github.com/gtxaspec/raptor/pull/97) (link Uranus via `URANUS_LIB`) and the runtime from the Magik 3.x kit (uClibc build; not shippable, like venus). **Then decide venus vs Uranus with the maintainer**; if venus stays ours, offer it as an alternative build of the same file, not a second backend. **ASK** before flashing.
- [ ] **Image needs for the Uranus path** (from the guide): `libstdc++`, `libgcc_s`, and a `librt.so.0` name for the uClibc build on uClibc-ng; L2 128K + `rmem=44M nmem=20M` (same as ours). They report L2 set from U-Boot left the camera off the network and a post-boot kernel module (#2093 method) worked — **our U-Boot `CONFIG_T41_L2_128K` worked on the v4 (2026-11-19)**; post that data point where it helps.
- [ ] **Richer models** (more COCO classes) once the base path works; the NNA has the headroom the T31 doesn't.
- **Gate:** the v4 produces the same `ivs-status` shape as the v3, so Phases 2–4 work unchanged on it.

## Phase 6 — benchmark + upstream

- [ ] **Benchmark against the G100** per [benchmark.md](benchmark.md): CPU, RSS, latency, and a detection-quality pass on the same frames. Record v3-JZDL vs v4-NNA vs G100.
- [ ] **Add andylee77's models to the benchmark** (repo `models/`, made with `tools/magik-train-txx`): seven T31 models (6 classes or person only, 4/8 bit) and one T41 NNA model (6 classes, ReLU6, 640x384, 42 ms). Their T31X figures to beat: 4-bit 448x256 0.18–0.4 s/frame (person mAP50 0.882), 8-bit exact (0.915) but ~4× slower; continuous upstream YOLOv5 left 4.5 % idle at 2 results/s.
- [ ] **PR resource stats** on every raptor/rhkd change (`tools/rhkd-stats.sh` rule): CPU/mem/flash tables.
- [ ] **Upstream the generic pieces:** rmd multi-class filter, the webui AI page, the raptor-hal JZDL multi-class decode and the NNA backend. Keep household/LAN specifics out.

## Phase 7 — sound detection (audio, parallel track)

Skill: `thingino-dev-persona`. A different modality from the video classes (audio, not frames), so it can proceed in parallel once Phase 2's event plumbing exists. The G100 does this with `CVoiceAlgo` on the mic PCM ([aqara-g100-detections.md](aqara-g100-detections.md)).

- [ ] **Audio classifier model** — small CNN on log-mel features, open data (AudioSet / ESC-50 subsets). Classes: baby crying, dog barking, glass break, smoke/CO alarm. Audio models are tiny, so plain CPU; export via `jzdl`/`magikconv` only if needed.
- [ ] **Feed from the `rad` audio path** (PCM), like the G100's `lumi_pushVoiceData`. Low rate, CPU-cheap, gated on sound energy.
- [ ] **Same event interface** → rmd/rhkd. HomeKit has no sound-class characteristic, so expose each sound as its own sensor (Occupancy/Motion or a programmable switch) and/or MQTT.
- [ ] **Web UI** — per-sound enable + sensitivity, alongside the video classes.
- **Gate:** a clap / alarm test triggers the right sensor; a quiet room stays silent. **ASK** before the on-camera run.

## Tools we build (inventory)

| Tool | Status | Where |
|---|---|---|
| `aitest` — on-camera CPU/RSS/latency harness | exists | [tools/aitest/](../tools/aitest/) |
| **Own JZDL runtime `ojzdl`** (open replacement for `libjzdl.m.so`) | **step 4 done 2026-11-24: yolov5 bit-exact, raptor-hal OJZDL backend identical to libjzdl on v3; PR #11.** (raptor-hal `src/hal_ivs_ojzdl.c`, `OJZDL=1`, now gtxaspec/raptor-hal#22; no libjzdl/libstdc++, 247 vs 244 ms end to end; yolov5 54/54 + 2/2 v3 bit-identical, 184–187 vs 175–183 ms.) Earlier: parity with libjzdl on v3 (84–90 vs 89–96 ms). (Step 1 bit-exact, scalar 2.84 s; step 3 MXU2 kernels for every layer type: persondet min 84–90 / avg 99–100 ms vs libjzdl 89–96 / 104–106, live daemons; 36/36 QEMU + 4/4 v3 bit-identical; arena 1.46 MB + scratch 0.54 MB, no per-frame mallocs.) Plain C bit-exact vs `jzdl_ref.py`/libjzdl under QEMU first, then MXU2 optimisation, then a raptor-hal drop-in. Gains = openness + ops libjzdl lacks, not size (libjzdl is 609 KB). Risk = speed on the single-core T31 (libjzdl ~100 ms persondet) | [tools/ojzdl/](../tools/ojzdl/), notes [jzdl-runtime.md](jzdl-runtime.md) |
| `jzdl/` — JZDL parser/writer/converter (v3 models) | in progress | [tools/jzdl/](../tools/jzdl/) — our edge over andylee77's `magik-train-txx`, which can only fine-tune the Magik 1.0 kit's sample network ("a float model cannot be converted for the T31") |
| QAT training pipeline (person/pet/package, open data) | exists: `train/` (QAT, exporter, `build_model.py`, model library); training runs on the teams' GPUs | `train/` |
| Benchmark runner + result log | partly (aitest) | [benchmark.md](benchmark.md) |
| webui AI plugin | written (preview overlay + Settings > Detection), in both images since 2026-11-26; not PR'd | thingino-firmware |

## Open decisions (ASK the owner)

- Detector in `rvd` or a separate daemon (Phase 0) — default in `rvd`.
- ~~One HomeKit Motion Sensor for "any class", or one sensor per class (Phase 2).~~ **One sensor per class (owner, 2026-11-25).**
- ~~Whether to draw boxes on the encoded stream/HKSV, or only in the web UI (Phase 3).~~ **Web UI only for now (owner, 2026-11-25).**
- ~~v4 flash + SD card for the NNA payload (Phase 5).~~ **Not needed: onna runs without nmem or an SD payload; the L2 fix is upstream.**

## Notes from the v4 runs (2026-11-25 / 26)

### v2 person threshold (2026-11-25)
v2 chain T41 file on onnaq -e cpu, full-val frames: person AP50 0.438 (executor int8 0.431, so onna costs nothing).
153 frames with 1-2 people >= 48 px, share of people found at IoU 0.5 / false person boxes per frame:
thr 0.40 = 24% / 0.04; 0.30 = 40% / 0.05; 0.20 = 52% / 0.12; 0.15 = 60% / 0.21; 0.10 = 66% / 0.41.
The v4 uses person 40; train.py model_conf THRESH person 40 (PR #57). Suggested: 20. Owner 2026-11-25: neither, keep 40 on the v4 and in model_conf.
Live v4: resize 20.8-23.2 ms, network 30.0-32.9 ms.

### v4 reset 2026-11-25 19:39:27 (timing run)
Standalone onnaq.mips (main c18b1e2 build) `-e nna -TT -n 20`, four models in order v2, yolon, yoloxn, mnv4n, rvd + watcher stopped. Reset ~35 s in (output in /tmp, lost). dmesg: CPU0 RESET ERROR PC 8007BAB8, CPU1 PC 80386478 (13:52 was 800C0CFC / 8001C088). Phase A at 14:05 (-T, yolon only, 7500 runs) did not reset. v4 restored 19:40:18 to the rvd/rmd/rhkd + yoloxn set (the image rvd then could not load yoloxn: "too many W blocks").
Correction (same day, 19:55): the 14:05 phase A ran the SAME onnaq.mips as 19:39. Engine code (nnaeng/nnacpu/be_mxu3) unchanged since 12:29; onnaq.c since then = stride-2/x64/dense on the NNA + vadd (2445800), graph-from-bin, NO_VADD; all in both runs. So 19:39 is NOT a regression: the reset is intermittent with the same code. Tally that day: v2 in rvd for hours, no reset; YOLO-N in rvd 55 min clean + 1 reset (13:52 after ~2.5 min); standalone YOLO-N 11 min clean + 1 reset (19:39 after ~35 s, v2 or yolon). Paths v2 may not use: raw readback (shift != 2), stride-2 pad copy. Next (needs owner OK + camera time): long soaks with env switches that keep raw / stride-2 convs on the CPU, output streamed off the camera.

### raptor-hal #25 live test (owner OK, 2026-11-25)
- v3 rvd with #25, yoloxn: 19:36:37-20:08:36 (32 min) clean, 58 vehicle detections, rvd 20-21 %, RSS 11.7 MB flat, RTSP 19.95 / 15.00 fps. Then 10 min v2 control (temp conf), then undo to the image rvd.
- v4 rvd with #25, yoloxn: 19:45:14-20:12:48 (27.5 min) clean, no reset, resize 14-17 ms (was ~21), network 74.6-83.8 ms, rvd 23-24 %, RSS 20.7 MB, RTSP 24.8 fps. One rvd re-exec 19:52:08 = a config reload (vehicle threshold 25). Restored afterwards. v2 control skipped.
- #25 now carries ONNA_NO_VADD (b2daafe). PR #75 adds ONNA_CPU_RAW / ONNA_CPU_S2 (not yet in #25).

### BUG: nnaeng.c wcache[256] overflow (2026-11-25 ~20:20)
`static uint8_t *wcache[256]` indexed by nd->rec, no bounds check. Max rec: v2 59, YOLO-N 216 (ok), yoloxn 323 (68 over), mnv4n 359 (104 over). In rvd (nm of the unstripped rvd) wcache is followed by onnaq's wtrim[MAXREC] -> yoloxn layers 256..323 run with wrong weights inside rvd; standalone onnaq.mips has nna_add_lut statics there (harmless, so standalone heads matched). Likely why yoloxn in rvd on the v4 reports 0 detections. Fix: PR #76 (cache MAXREC-sized + range check); raptor-hal #25 needs the same before rebuilding. Not the YOLO-N reset cause (217 recs fit).
- #25 v3 PASS: v2 control 20:09-20:19 clean (rvd 17 %, 11.6 MB); #25 vs image rvd on yoloxn 20-21 % vs 22 %, RSS equal. v3 restored 20:20:42 (image rvd, no binds). ojzdl has no fixed per-layer table (v3 unaffected by the wcache bug). v4 #25: stable, but yoloxn results invalid until the wcache fix; re-run after it.

### BUG?: rvd death after `save raw` + `save jpeg` on the v4 (2026-11-25 20:27)
- After the wcache fix: v4 rvd = #25 with the fix + the raptor class branch, bind-mounted at 20:26:53 (owner OK). Daylight check: SUV = vehicle 0.42-0.45, PC 0.44 (fix confirmed). **The owner keeps it running** (the previous rvd kept beside it).
- rvd died ~20:27:32-33, about 1 s after `raptorctl rvd save raw /tmp/f2.nv12 1` + `save jpeg /tmp/f2.jpg 1` at 20:27:31. Last rvd log line: "jpeg chn 3: stopped (no consumers)" 20:27:32. No further rvd line, no kernel line (no OOM, no segfault print). ric "RVD query failed (-146)" from 20:27:34; watcher "rvd died, restarting all" 20:27:36; new rvd stable (25 fps, 23 %, 21.5 MB). Not a camera reset.
- Same boot, earlier: dmesg `do_page_fault(): sending SIGSEGV to rvd for invalid read access from 00000068` at uptime 80.96 s = 20:25:35. That's before the new rvd install, during the restore of the previous set after the 20:23 watchdog reset; logread no longer covers it. (20:23 reset PCs: CPU0 800A396C, CPU1 803864B0.)
- The same two saves on the previous rvd at 19:41 did NOT kill rvd.
- **TODO (only when someone has the v4 free, owner OK, never two NNA users):** repeat `save jpeg` (then `save raw`) on both rvd builds, with logread + dmesg captured right after; decide save path (jpeg chn 3 start/stop, rvd_ctrl.c) vs onna (#25) vs chance. No saves until then. An unstripped rvd build maps a fault PC if one is printed.

### v4 resets: memory pressure lead (2026-11-25 ~20:40)
Reset PCs mapped with the System.map of a kernel build from the same commit as the image (addresses very likely identical; the camera has no /proc/kallsyms):
- 13:52 YOLO-N in rvd: CPU0 list_lru_count_one+0x10 (shrinker / reclaim) | CPU1 ingenic_wait_irqoff+0x24
- 19:39 standalone: CPU0 rcu_idle_exit+0x28 | CPU1 _raw_spin_unlock+0x24
- 20:22 standalone yolov5n: CPU0 oom_scan_process_thread+0x10 (OOM killer) | CPU1 _raw_spin_unlock_irq+0x18
Two of three have CPU0 in memory reclaim / OOM code. v4 then (yoloxn rvd 21.6 MB): MemAvailable ~10 MB, no swap, min_free_kbytes 980. Standalone yolov5n peaks 25-29 MB with only ~31 MB available after rvd stops. Hypothesis: low memory -> reclaim thrash (squashfs code pages) -> the watchdog daemon (-T 60) not fed -> reset; not (only) an NNA deadlock. To try: log MemAvailable through a reset (overnight watch), shrink onna's footprint (free add tables once vadd is verified, tensor reuse, per-rec weight caches), standalone runs only with a memory check first.

### 2026-11-26 05:31: v4 reset while copying an image into /tmp (memory lead, 3rd OOM-path PC)
- With rvd (yoloxn) running, MemAvailable ~8 MB; `cat > /tmp/fw.bin` (16 MB tmpfs) reset the camera. PCs: CPU0 ingenic_wait_irqoff+0x24, CPU1 oom_scan_process_thread+0x68. Second copy with rhkd/watcher/rvd/rad stopped (MemAvailable 37 MB) went through.
- Both cameras flashed 05:33 (v4 20261125-2114) / 05:38 (v3 20261125-2115); image-native rvd on both, no binds. Log-only reset watch on the v4 until 08:00 (min MemAvailable logged).

### 2026-11-26 06:05: onnaq memory pool on the v4
- #77 (tensor pool, weights dropped after frame 1) + #78 (SiLU 32-ch fix), verified standalone on the v4 05:47-06:00.
- raptor-hal: gtxaspec/raptor-hal#25 fast-forwarded 7cbbce9 → eae6739 (src/onna = 2f90bf0) + comment (owner OK 2026-11-26), merged into our HAL integration branch.
- rvd with it bind-mounted on the v4 06:05: rvd RSS 21.9 → 13.4 MB, MemAvailable 11.9 → 19.8 MB, yoloxn 87 ms. 30-min soak logged.
- 06:28: #79 (ojzdl weights held once, model buffer dropped; T31 yoloxn heap 4.23 → 2.63 MiB) → gtxaspec/raptor-hal#22 fast-forwarded 6da129d → 0fa78eb + comment (owner OK 2026-11-26), also in our HAL integration branch.
- 06:47: v3 A/B of ojzdl #79 (owner OK): rvd RSS 11.7 → 10.2 MB, CPU 20-21 % both, detections in both; v3 back on the image rvd. 06:55: this HAL is the standard for our image builds; the standard builds are the tested binaries, and the per-camera must-have checks pass.
