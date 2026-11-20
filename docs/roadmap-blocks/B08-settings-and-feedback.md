# B08 — Settings and feedback

**Status:** ⬜ not started (draft card: criteria to be agreed with the owner)
**Weight:** 3 (from `docs/03-ROADMAP.md`)
**Decisions this block rests on:** "Fixed pipeline settings: batch 1, LAB color correction, `sdpa`, no `torch.compile`";
"Interruption checks only `shared.state.interrupted`"; "BlockSwap is a manual slider with a note for 7B on small GPUs";
"VAE tile 512 is the default"; "New script arguments go at the end of `script_args`"; "API values of the numeric
settings are clamped to the slider ranges".

## Why this block
Today a SeedVR2 run is a black box: no progress in Forge's bar, Interrupt waits for the whole DiT pass, color
correction and attention are fixed, and an owner of a 12 or 24 GB GPU has to guess tile size and BlockSwap. After the
block the run is visible and stoppable, and the main settings are one choice away.

## What we do (steps)
1. Forge progress bar: map the four phases (encode / DiT / decode / postprocess) and their batches onto
   `shared.state` (per the fact-check), in `seedvr2_neo/` through `HostHooks`, no Forge import there.
2. Interrupt inside the DiT step: a forward pre-hook on each DiT block (`self.blocks`, 32 in 3B, 36 in 7B) that raises
   when `shared.state.interrupted`; registered for the run and always removed (cached models keep no hook). Plus
   `ctx['interrupt_fn']` for the checks at the top of each batch. No `seedvr2_vendor/` patch.
3. Color correction choice under "Advanced Settings": `lab` (default) / `wavelet` / `wavelet_adaptive` / `hsv` /
   `adain` / `none`; written to the infotext; API value validated (unknown → `lab`).
4. Attention mode under "Advanced Settings": `sdpa` (default) plus only the backends whose packages import
   (`compatibility.py` flags); an unavailable API value falls back to `sdpa` with a console note.
5. New title without "(24G)" (wording to agree) — with the compatibility consequences from the fact-check (API
   `script_name`, ui-config keys, elem_ids).
6. GPU profiles 8 / 12 / 24 GB: a dropdown that fills VAE tile size (and overlap) and BlockSwap; the sliders stay
   visible and editable; "Custom" when the user moves a slider. The profile itself is not a script argument.
7. Tests for each step in the mirrored folders; docs (`02-ARCHITECTURE`, `05-DECISIONS`, `09-PROCESSES`, README).
8. Live check on the 8 GB GPU through the Forge API: progress moves, Interrupt mid-DiT frees VRAM within seconds,
   each color correction gives an image.

## Acceptance criteria
- [ ] During a run Forge's progress bar moves through the phases (live screenshot / `/internal/progress` polled).
- [ ] Interrupt pressed during the DiT phase stops the run before the phase ends (live: time from Interrupt to the
      end of the run is seconds, not the full DiT time); VRAM is released; inputs come back un-upscaled.
- [ ] No hook stays on a cached DiT after a run, a failure or an interrupt (test).
- [ ] Color correction and attention mode are selectable, saved in the infotext, and API calls without them keep the
      old behaviour (`lab`, `sdpa`) (tests).
- [ ] The Script title has no "(24G)"; the compatibility effects found by the fact-check are handled or documented.
- [ ] Choosing a GPU profile changes the tile and BlockSwap sliders; the sliders still change freely (test of the
      profile table; live look at the UI).
- [ ] All tests pass (`python -m pytest tests`), CI green.

## Which checks are needed on `/block-done`
Always: `code-auditor`, `fresh-eyes-reviewer`, `plan-drift-checker`. Additionally:
- [x] `integration-checker` — new script arguments, the block must run in Forge
- [ ] `security-auditor` — no user data, files or external requests touched
- [x] `stress-tester` — API inputs (unknown color correction / attention values, wrong types)

## Dependencies
B11 closed (UI layout). Live checks: Claude launches the owner's Forge with `--api`.

## Out of scope for this block
- DiT tiling, 6K/8K (B12).
- Automatic BlockSwap from detected VRAM (decided: manual; profiles only fill the sliders when chosen).
- `torch.compile`, fp16 compute (`06-IDEAS.md`).
- Settings in Forge's Settings tab (`06-IDEAS.md`).

## Hidden assumptions
- `check_interrupt(ctx)` in the vendored phases runs only at the top of each batch loop, the same granularity as
  `progress_callback`; with batch 1 the DiT phase is one batch, so a real mid-DiT stop needs the block hook (step 2).
- BlockSwap wraps the DiT blocks' forward; the hook must work with and without BlockSwap (to check in tests and live).
- Progress (fact-check, `external-apis/forge-neo.md`): the bar is `job_no/job_count + sampling_step/sampling_steps /
  job_count`; after `process_images` it reads 100%, so we must write the fields ourselves. Writing `job_count` breaks
  img2img Batch's outer bar and turns Interrupt into a soft stop when `job_count > 1` — so the plan is to move only
  `sampling_step` / `sampling_steps` (and `textinfo`), never `job_count`; no `state.begin()/end()`.
- Title (fact-check): ui-config control values survive (keyed by file name); API `script_name` with the old title gets
  HTTP 422; the saved "Script" dropdown choice and pasted `Script:` infotext stop selecting it; the generated
  `elem_id`s change (ours: the refresh button only).

## Open questions
See the report on `/go`.
