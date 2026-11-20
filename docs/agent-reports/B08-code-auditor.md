# B08 — code audit

Scope: `git diff main...feature/b08-settings-feedback` (committed, HEAD `9ca1c9d`), plus a look at the uncommitted
working-tree change to `scripts/seedvr2_script.py` / `tests/scripts/test_seedvr2_script_settings.py` that appeared
during the audit (`finish_forge_progress`). Rules: `docs/process/code-style.md`, `CLAUDE.md`.
Forge facts checked against the local `neo` checkout (`modules/img2img.py`, `modules/processing.py`,
`modules/shared_state.py`, `modules/progress.py`).

Targeted tests: `test_run_progress.py`, `test_upscale_run.py`, `test_pipeline_options.py`, `test_gpu_profiles.py`,
`test_seedvr2_script_settings.py`: 117 passed. A probe test (scratchpad, not in the repo) ran three img2img Batch
files through the script stubs.

Summary: BLOCKER 0, IMPORTANT 1, MINOR 8.

## Checked and found correct (no finding)
- **Hooks always removed:** `watching_modules` removes every handle in `finally`, including when registration fails
  part-way; the hook returns None (no input replacement). Cached models carry no hook after success / error / interrupt
  (test covers all three). On interrupt the models are dropped anyway (`ModelLifecycle.finish_run`, `succeeded=False`).
- **Hooks reach the modules that run:** `materialize_model` loads with `load_state_dict(..., assign=True)`, GGUF
  replaces only Linear layers (no children, not watched), `CompatibleDiT` wraps and keeps `dit_model`; BlockSwap
  replaces `block.forward`, but pre-hooks run in `Module.__call__` before it; the NaDiT loop calls `block(...)` via
  `gradient_checkpointing` → `module(...)`. An interrupt from a child container of a swapped block leaves that block
  on the GPU, but the failed run drops the models.
- **Blocks per image:** the vendor forces `cfg.scale = 1.0` and `steps = 1` (`generation_phases.py:599-601`), so one
  DiT forward per image; `len(blocks)` is right, and the cap at one image covers a future guidance pass.
- **Progress monotonicity inside a run:** `start_phase` at the end of the previous phase's weight; `batch_done` takes
  `max`; block share is capped per image and replaced by the batch share. Vendor callbacks are
  `(current, total, frames, name)` with `total` = images for batch size 1.
- **No vendor `except Exception` swallows `UpscaleInterrupted`:** the phase handlers and `CompatibleDiT.forward`
  log and re-raise.
- **API positional args:** new values appended at the end; 13 / 16 / 19 / 21 positional calls work (19 tested).
- **Hostile values:** `parse_color_correction` / `parse_attention_mode` (None, int, list, "", unknown) → defaults;
  `find_profile` with None / unknown → Manual behaviour; `RunProgress.batch_done` ignores unreadable values.
- **Forge state:** txt2img (incl. Hires. fix: `sample_hr_pass` calls `nextjob()`, `processing.py:1514`) ends with
  `job_no == job_count`, so the upscale becomes job 1 of 1; `begin()`'s `-1` is handled; `sampling_steps` left at 1000
  is harmless (the sampler sets it before `total_tqdm` reads it).
- No block numbers, TODOs or debug prints in the new code and comments.

## IMPORTANT

### 1. img2img Batch: the progress bar restarts at 0 for every file — 🟢 R2 (S1×L2)
`scripts/seedvr2_script.py:122-131` (HEAD `start_forge_progress`); test
`tests/scripts/test_seedvr2_script_settings.py:46-52` (HEAD).

What is wrong: the code assumes "in img2img Batch Forge counts the files and its counter stays". Forge sets
`state.job_count = len(batch_images) * p.n_iter` once (`modules/img2img.py:46`) and never advances `job_no` in the
batch loop: only `process_images_inner` calls `state.nextjob()` (`processing.py:1018`), and this script never runs
`process_images` for img2img. So `job_no` stays 0 for every file.

Scenario: img2img Batch of 10 files → for each file the bar goes 0 → 10 % and jumps back to 0 at the next file;
Forge's ETA (`elapsed / progress - elapsed`) at file 5 halfway is ~85× the per-file time instead of ~5.5×. The
committed test seeds `job_no = 1`, a state Forge never produces for this script, so it passes with the bug.

Status: the uncommitted working-tree change (`finish_forge_progress`, advancing `job_no` by `n_iter` after each
file, plus a three-file test starting at `job_no = 0`) fixes it; a probe run of three files gives per-file ranges
0.04–0.33, 0.37–0.67, 0.71–1.0 and ends at 1.0. Commit it with the test; update the docs (MINOR 3).

## MINOR

### 2. img2img Batch with Batch count > 1: each file's upscale fills only its first job slot — 🟢 R1 (S1×L1)
`scripts/seedvr2_script.py:122-146` (working tree, `start_forge_progress` / `finish_forge_progress`).
Scenario: 2 files, Batch count 2 → `job_count = 4`; file 1's upscale moves the bar 0 → 25 %, then
`finish_forge_progress` jumps it to 50 %; same for file 2 (50 → 75 → 100 %). Fix: when Forge counts the jobs, set
`sampling_steps = PROGRESS_STEPS // n_iter` (Forge's formula clamps only the total, not `sampling_step / sampling_steps`), so
one file spans all of its `n_iter` slots.

### 3. Docs still say Forge's counter "stays" in img2img Batch — 🟢 R1 (S1×L1)
`docs/05-DECISIONS.md:177-179`, `docs/09-PROCESSES.md:131-132`. With the fix the script advances `job_no` itself
past each file (Forge leaves it to `process_images`). A reader following the decision would remove
`finish_forge_progress` as redundant. Fix: rewrite both paragraphs (and add the fact "the img2img Batch loop does not
advance `job_no`" to `external-apis/forge-neo.md`, progress section).

### 4. Attention fallback differs from the vendor's chain — 🟢 R2 (S1×L2)
`seedvr2_neo/pipeline_options.py:79-90` (`resolve_attention_mode`).
The vendor's `validate_attention_mode` (`seedvr2_vendor/optimization/compatibility.py:175-260`) falls back
`flash_attn_3 → flash_attn_2 → sdpa` and `sageattn_3 → sageattn_2 → sdpa`; ours goes straight to `sdpa`.
Scenario: RTX 40xx with `sageattention` 2 and the `sageattn3` package installed (listed in the UI) but SA3 failing to
import (non-Blackwell) → user picks `sageattn_3` → the run uses `sdpa` (slower) instead of `sageattn_2`; same for an
API call with `flash_attn_3` where FA2 is available. Fix: mirror the vendor's two-step chain in
`resolve_attention_mode` (with a test), so the infotext still names the backend actually used.

### 5. Enum parsing duplicated — 🟢 R1 (S1×L1)
`seedvr2_neo/pipeline_options.py:94-102` (`_parse_choice`) repeats `seedvr2_neo/upscale_settings.py:153-162`
(`parse_size_mode`); the copies already differ: `_parse_choice` compares `member.value == wanted` without casefolding the
member. Scenario: a future member with an upper-case value (like the size modes, "Shortest edge") added to
`ColorCorrection` / `AttentionMode` would never match its own UI string and silently fall back to the default. Fix:
one public `parse_enum_choice(value, enum, default)` (casefolding both sides) used by both modules.

### 6. Phase indexes defined in two modules — 🟢 R1 (S1×L1)
`seedvr2_neo/upscale_run.py:32` (`ENCODE_PHASE, DECODE_PHASE, POSTPROCESS_PHASE = 0, 2, 3`) vs
`seedvr2_neo/run_progress.py:23-24` (`PHASES`, `UPSCALE_PHASE = 1`). Scenario: a phase inserted or reordered in
`PHASES` (e.g. a separate "Loading" phase) leaves `upscale_run` publishing wrong phase names and weights; no test ties
the two lists. Fix: define all four indexes (an `IntEnum`) next to `PHASES` and import them.

### 7. Interrupt inside the DiT prints ERROR lines that look like a crash — 🟢 R3 (S1×L3)
`seedvr2_neo/upscale_run.py:189-193` raising inside the forward; logged by
`seedvr2_vendor/optimization/compatibility.py:929-937` (`CompatibleDiT.forward`: "Forward pass error: User
interrupted generation." at ERROR, plus a dtype info line, `force=True`) and `generation_phases.py:760-762`
("Error in Phase 2 (Upscaling)"). Every DiT is wrapped in `CompatibleDiT`, so every Interrupt during the DiT step
prints them before our "Process Interrupted by User". Fix: add a line to `07-RISKS.md` / README (no vendor patch needed);
optionally print "the errors above are the interrupt" in the `UpscaleInterrupted` handler.

### 8. `_upscale_tensor` grew to ~86 lines with two abstraction levels — 🟢 R1 (S1×L1)
`seedvr2_neo/upscale_run.py:248-333`. Context and runner setup plus the phase sequence with hook, progress and
memory-release bookkeeping in one method (guideline 30–40 lines). Fix: extract the `with watching_modules(...)` part
into `_run_phases(runner, ctx, frames, target, request, debug, debug_mode)`.

### 9. New public functions without type hints — 🟢 R1 (S1×L1)
`scripts/seedvr2_script.py:189` `apply_gpu_profile(profile_label, dit_model_name)`, `:200`
`resolve_pipeline_options(pipeline, color_correction, attention_mode)`, `:137` `finish_forge_progress(..., iterations)`;
`seedvr2_neo/upscale_run.py:129` `runner_dit(runner)`, `:135` `dit_block_types(runner)`, `:184` `phase_progress`,
`:189` `module_called(self, module)`; `seedvr2_neo/run_progress.py:57` `batch_done(self, current, total)`.
`code-style.md` asks for hints on all public functions (`Any` at the API boundary). Fix: add them.
