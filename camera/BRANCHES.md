# Mirrored branches

The Thingino integration code (raptor, raptor-hal, rhkd, web UI) is **developed on the upstream forks**, not here — that's how it gets PR'd upstream. This repo mirrors it as patches under `camera/patches/`. This file records, for each change, the exact fork branch and commit the patch was generated from, so the mirror is regenerable and the upstream path stays intact.

**Rules**
- Develop on the fork branch; regenerate the patch here (`git format-patch` / `git diff`), never hand-edit `camera/patches/*`.
- Keep everything public-safe: no IP addresses, host names, local paths or household details.
- When a change merges upstream, drop its row (upstream carries it) and remove the patch.

| Change | Upstream repo | Fork branch | Commit | Patch | PR | Status |
|---|---|---|---|---|---|---|
| raptor-hal: multi-class JZDL decode — generalise `hal_ivs_jzdl.cpp` to `3×(5+N)` | gtxaspec/raptor-hal | _tbd_ | — | — | — | not started |
| raptor: `rmd` per-class filter + split notify/record | gtxaspec/raptor | _tbd_ | — | — | — | not started |
| raptor: `ivs-status` carries class + box + track_id | gtxaspec/raptor | _tbd_ | — | — | — | not started |
| webui: AI detection page (enable/disable, per-class settings, live box overlay) | themactep/thingino-firmware | _tbd_ | — | — | — | not started |
| raptor-hal/raptor: NNA (venus) backend behind the same interface | gtxaspec/raptor-hal, gtxaspec/raptor | _tbd_ | — | — | — | not started (overlaps raptor-hal#21 Uranus — coordinate) |
| rhkd: sound-event sensors (baby cry, dog bark, alarm) | Small6oy/thingino-homekit | _tbd_ | — | — | — | not started |

PR status, once these open, is tracked in the private research notes' upstream tracker.
