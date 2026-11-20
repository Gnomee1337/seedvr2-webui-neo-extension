# Fresh-eyes review — B11 output size (`git diff main...HEAD`, branch `feature/b11-output-size`)

## How I understood the result

The SeedVR2 script gets a size group: a radio (Scale factor / Shortest edge / Long edge) that shows one control at a
time, quick ×1.5/×2/×4 buttons, and a Markdown line "Output size: W×H from w×h". `UpscaleSettings.target_for(size)`
turns the mode into the pipeline's `resolution` / `max_resolution` per input size (long edge capped at 3840 in the new
modes, shortest edge raised to 512); the run, the pass budget, the console line, the infotext and the preview all use
it. txt2img reads Forge's width/height/Hires. fix components (collected in `after_component`); img2img reads the image
size in the browser and a 1 s JS poll presses a hidden refresh button. Three new positional args are appended to
`run()` with defaults that mean "Shortest edge", so old API callers keep their size; the UI radio is not saved to
`ui-config.json` and always starts on Scale factor ×2. Seed / Save original / Force reload / Debug moved into
"Advanced Settings". This matches the task; nothing extra beyond the regroup and docs.

Checked against the vendored resize (`seedvr2_vendor/data/image/transforms/side_resize.py`, `core/generation_utils.py`
even-trim): `OutputTarget.output_size` reproduces it (shortest edge → `resolution`, long edge truncated, `round` scale
when over `max_resolution`, both sides `//2*2`). Shortest-edge mode gives the same `resolution` / `max_resolution` as
before, so old API callers' outputs are unchanged.

## IMPORTANT

### 1. The txt2img size line starts from code defaults, not from the values the user actually sees
`scripts/seedvr2_script.py:222-225` builds the initial Markdown from `component.value` inside `ui()`. Per
`docs/external-apis/forge-neo.md:181-183` (and the author's own comment at `scripts/seedvr2_script.py:203-205`),
`ui-config.json` values are applied by `UiLoadsave.add_block` *after* the scripts' UI is built. Nothing re-computes the
line on page load (only `.change` events, which do not fire for initial values).
Scenario: a user has set txt2img defaults to 832×1216 with Hires. fix on ×1.5 (Settings → Defaults), or saved Scale
Factor 1.5. The page opens showing "Output size: **2048×2048** from 1024×1024", while Generate produces 2496×3648
(or the ×1.5 result). The line is wrong on every page load until a control is touched — exactly the "shown ≠ produced"
case the line exists to prevent. Possible fix direction: in `javascript/seedvr2.js`, press a hidden txt2img refresh
button once on `onUiLoaded` (the img2img side already does this through its first poll tick).
Rating: 🟡 R4 (S2×L2).

### 2. Existing UI users silently get a much bigger (and slower) output on their first run after the update
`seedvr2_neo/upscale_settings.py:52` (`DEFAULT_SIZE_MODE = SCALE_FACTOR`), `SCALE_FACTOR` default 2.0, plus
`scripts/seedvr2_script.py:206` (mode never saved). The old default was 1080 px shortest edge. With Neo's default
1024×1024: old 1080×1080 (1.17 MP) → new 2048×2048 (4.2 MP, ×3.6 pixels). With Hires. fix ×1.5 (common for SDXL):
1536² → 3072×3072 = 9.4 MP, more than 3840×2160 (8.3 MP), which README/RISKS measure at ~8 min (no `--cuda-malloc`) to
~17.5 min (with it) with VRAM full on the 8 GB target; a square 9.4 MP may not even fit. A user who saved Shortest Edge
1440 also loses it: every launch shows Scale factor, so unless they notice the radio each session they get ×2.
`docs/07-RISKS.md:112` says "×3–×4 of a txt2img image after Hires. fix lands there easily" — the *default* ×2 after a
×1.5 Hires. fix already lands there, so the entry understates likelihood. The owner chose ×2, so this is not a code
defect, but the consequence (default run time on 8 GB jumps from ~1 min to many minutes, possible OOM) is not stated
anywhere a user reads before pressing Generate; README only lists the 4K timing under "Large outputs take minutes".
Suggest: correct the RISKS entry (default + Hires. fix), mention in README/changelog that the default output changed,
and consider whether a remembered mode is worth more than the radio/slider mismatch argument.
Rating: 🟡 R6 (S2×L3).

## MINOR

### 3. "Forge passes a shorter script_args list as is" holds only when our script's slice is the last one
`scripts/seedvr2_script.py:336-338` (comment), `docs/05-DECISIONS.md` (new entry "New script arguments go at the end…"),
README "API" section ("The last three arguments may be left out"). Per `forge-neo.md:185-187`, `init_script_args` does
`script_args[args_from:args_to] = request.script_args`. A shorter right-hand side *shrinks* the list, and the runner
then reads `args[args_from:args_to]` — 19 items, the last 3 taken from whatever script's args follow ours (e.g. a
selectable script of an extension loaded after this one, like Ultimate SD Upscale). The live "TypeError on a short list"
proves our slice was last on the test install, not in general. In practice `size_mode` gets an unrelated value and
falls back to Shortest edge, so the size is still right — but the comment and the decision state a mechanism that is
not the one at work, and a following script whose first default is `"Long edge"`/`"Scale factor"` (or a list/number
parse oddity) would change the size. Suggest rewording to "an unknown mode falls back to Shortest edge, which also
covers the shifted values Forge passes when the list is short", and verifying with a second selectable script installed.
Rating: 🟢 R2 (S2×L1).

### 4. `/sdapi/v1/script-info` advertises a default the API does not use
`api_info` is built from the controls' code values (`forge-neo.md:181-184`), so script-info shows `size_mode` =
"Scale factor", scale 2.0; an API caller that omits the args gets "Shortest edge". A client that builds `script_args`
from script-info defaults (a common pattern) gets ×2 instead of the 1080 px it got before. README explains the omission
rule but not this discrepancy.
Rating: 🟢 R2 (S1×L2).

### 5. "Raised to the 512 px minimum" / "no input is upscaled below 512 px" is false for inputs thinner than 4:1
`seedvr2_neo/upscale_settings.py:115-119`, `seedvr2_neo/size_preview.py:30`, radio info at
`scripts/seedvr2_script.py:201-202`, README ("no image is made smaller than 512 px on its shortest edge").
Computed: 100×1000, ×2 → `OutputTarget(512, 2048, raised=True)` → output **204×2048**, and the line says
"(raised to the 512 px minimum)". 512×4096 in Long edge 2048 → 256×2048. Also, without any hint, ×2 of 400×2000 gives
614×3072 (×1.54, the old 4:1 cap) and ×2 of 3000×2000 gives 3840×2560 (×1.28, the 3840 cap). The numbers in the line are
right (same math as the run), but the wording promises a minimum it does not keep, and a capped factor is not flagged.
Rating: 🟢 R2 (S1×L2).

### 6. Forge's txt2img sliders are wired on `.change`, not `.release`
`scripts/seedvr2_script.py:233-235`. Each drag tick of Width/Height/Upscale by sends a server request (Neo's own Hires.
preview uses `.release`, `forge-neo.md` "Neo already shows this value"). Harmless but chatty; it also queues requests
during a long generation.
Rating: 🟢 R2 (S1×L2).

### 7. Unverified: the img2img browser size equals the image the script receives
`javascript/seedvr2.js:7-10` reads `naturalWidth`/`width` of the first visible `img`/`canvas`; the run upscales
`p.init_images` decoded from ForgeCanvas's background PNG. If ForgeCanvas stores a resized copy of a large upload, or the
Sketch tab's first element is a canvas sized to the editor, the line and the result differ. Neo's own "Resize by"
preview shares the assumption, which lowers but does not remove the risk. Live check: a 5000+ px upload and the Sketch /
Inpaint tabs, compare the line with the console `[SeedVR2] … ->` line.
Rating: 🟢 R2 (S2×L1).

### 8. Generic note reads oddly in txt2img
`seedvr2_neo/size_preview.py:17` — when Forge's controls were not found (`ui_reorder_list` puts scripts first), txt2img
shows "Output size: depends on each input image"; txt2img has no input image. A reader/user will wonder which image.
Rating: 🟢 R1 (S1×L1).

### 9. A saved fixed seed is now hidden in a collapsed accordion
`scripts/seedvr2_script.py` Advanced Settings row: Seed moved out of the main view. A user with a non-`-1` seed saved in
`ui-config.json` no longer sees that every run reuses it. Part of the requested regroup; noting the side effect only.
Rating: 🟢 R1 (S1×L1).

### 10. Two copies of the "raised" message
`seedvr2_neo/upscale_run.py:104` uses `target.resolution`, `seedvr2_neo/size_preview.py:30` hard-codes "512"; one uses
`x`, the other `×`. Same fact in two places; a change of the minimum updates one of them.
Rating: 🟢 R1 (S1×L1).

## Checked, no finding
- Shortest-edge mode (old API callers, 13–16 args) → identical `resolution`/`max_resolution` to `main`; the pass
  budget now uses the capped output size (more accurate, not a behaviour change visible to users).
- Per-image infotext: params list built from `collected.images`, same length as `merge_run_images`, so `zip` drops
  nothing; failed images keep the per-image keys plus `SeedVR2 Error`.
- Hidden img2img Numbers/Button are not in the returned list, so `script_args` positions are unchanged.
- `do_not_save_to_config` is honoured by `apply_field` per `forge-neo.md:175-176`; radio and visible column agree on load.
