# B08 — Settings and feedback

**Status:** 🔄 in progress — code, tests, docs and live checks done; waiting for `/block-done`
**Weight:** 3 (from `docs/03-ROADMAP.md`)
**Decisions this block rests on:** "Fixed pipeline settings: batch 1, no `torch.compile`"; "Interruption checks only
`shared.state.interrupted`"; "BlockSwap is a manual slider with a note for 7B on small GPUs"; "VAE tile 512 is the
default"; "New script arguments go at the end of `script_args`"; "API values of the numeric settings are clamped to
the slider ranges". New in this block: the title, hooks on container modules, the progress bar, color correction and
attention, GPU profiles (`05-DECISIONS.md`).

## Why this block
Today a SeedVR2 run is a black box: no progress in Forge's bar, Interrupt waits for the whole DiT pass, color
correction and attention are fixed, and an owner of a 12 or 24 GB GPU has to guess tile size and BlockSwap. After the
block the run is visible and stoppable, and the main settings are one choice away.

## What we do (steps)
1. ✅ Forge progress bar: `seedvr2_neo/run_progress.py` (passes, weighted phases, batches, DiT blocks) → `HostHooks.report_progress`
   → `sampling_step` / `textinfo`; the upscale is job 1 of 1 unless Forge is inside its own job (img2img Batch).
2. ✅ Interrupt inside a phase: forward pre-hooks on the container modules of the VAE and the DiT (set before the
   phases, always removed) plus `ctx['interrupt_fn']`. No `seedvr2_vendor/` patch. A process-wide hook was tried
   first and dropped (it runs before every layer).
3. ✅ Color correction choice (6 methods, `lab` default), in the infotext; unknown API value → `lab`.
4. ✅ Attention choice: `sdpa` plus installed backends; a backend that does not import → `sdpa` with a console note;
   the infotext names the backend used.
5. ✅ Title "SeedVR2 Upscaler"; README tells API users the new name (old one → HTTP 422).
6. ✅ GPU profiles Manual / 8 / 12 / 24 GB fill BlockSwap (by 3B / 7B), tile size and overlap; moving a slider → Manual;
   not a script argument, not saved.
7. ✅ Tests (mirrored folders) and docs (`02-ARCHITECTURE`, `05-DECISIONS`, `07-RISKS`, `08-GLOSSARY`, `09-PROCESSES`,
   `external-apis/seedvr2-upstream.md`, README, `06-IDEAS`).
8. ✅ Live check on the 8 GB GPU through the Forge API (3B GGUF Q6_K, `--cuda-malloc`).

## Acceptance criteria
- [x] During a run Forge's progress bar moves through the phases: `/internal/progress` polled — 0 → 0.01 (encode)
      → 0.128 → 0.422 (inside the DiT step) → 0.598 → 0.951 → done.
- [x] Interrupt pressed during the DiT phase stops the run before the phase ends: 3.7 s from Interrupt to the end
      (models unloaded, VRAM 3.6 → 0.8 GB); during the decode 0.6 s (7.9 → 0.8 GB); during the encode 1.2 s; the
      inputs come back with `SeedVR2 Error: Interrupted`.
- [x] No hook stays on the models after a run, a failure or an interrupt (`test_no_hook_is_left_on_the_models_…`).
- [x] Color correction and attention mode are selectable, saved in the infotext, and API calls without them keep the
      old behaviour (`lab`, `sdpa`) (tests; live run of every color correction).
- [x] The Script title has no "(24G)"; the compatibility effects are documented (README, `07-RISKS.md`).
- [x] Choosing a GPU profile changes the tile and BlockSwap sliders; the sliders still change freely (tests). The
      owner's look at the UI is still open.
- [x] All tests pass (`python -m pytest tests`): 2583 passed, 1 skipped, 3 xfailed. CI: on push.

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
- Progress per VAE tile (the vendored decode reports nothing per tile; `07-RISKS.md`).

## Hidden assumptions
- The vendored loaders keep module objects (`assign=True`; GGUF swaps only Linear layers), so hooks set before a phase
  fire inside it — confirmed live (Interrupt mid-DiT and mid-decode, per-block progress). An upstream sync may break it
  (`07-RISKS.md`).
- A forward pre-hook that returns a value replaces the module's inputs; ours always returns None (found by a test).
- The slow decode with `--cuda-malloc` (~2.5–3 min for 1024×768) is the same on `main` (220.8 s vs 192.6 s with B08):
  not caused by the hooks (`07-RISKS.md` R6).

## Open questions
- The owner's look at the UI (GPU profile next to BlockSwap, the two new dropdowns in Advanced Settings).
