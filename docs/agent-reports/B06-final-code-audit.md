# B06 final code audit — VRAM/RAM lifecycle, BlockSwap

Scope: `git diff main...HEAD` on `feature/b06-vram-lifecycle` (`54af56e`..`30775da`): `scripts/seedvr2_script.py`,
`seedvr2_neo/{memory_plan,loaded_models,model_lifecycle}.py`, the local patch in
`seedvr2_vendor/optimization/memory_manager.py`, `tests/seedvr2_neo/test_*.py`, README and docs.
Checked against the vendored `core/model_cache.py`, `core/model_configuration.py` (`_initialize_cache_context`,
`_acquire_runner`, `_setup_dit_model` / `_setup_vae_model`), `core/generation_phases.py` (where models are cached and
cleaned up), `core/generation_utils.py`, `optimization/compatibility.py`.
Findings of `B06-code-audit.md` and `B06-fresh-eyes.md` that were fixed or moved to `07-RISKS.md` are not repeated.

Tests: `329 passed, 1 skipped, 2 xfailed` (`uv run --no-project --python 3.10 --with-requirements requirements-dev.txt python -m pytest tests -q`).

Summary: **0 BLOCKER, 0 IMPORTANT, 9 MINOR.** The code logic holds in every path I traced; what is left is docs that
still describe the pre-B06 state, two open leftovers from earlier rounds, and clean-code nits.

---

## MINOR

### 1. `02-ARCHITECTURE.md` still describes the old runtime — 🟢 R2 (S1×L2)
`docs/02-ARCHITECTURE.md:16`: "single CUDA GPU (`cuda:0` is hard-coded)"; `docs/02-ARCHITECTURE.md:114`: "In-memory
caches | `GlobalModelCache` (models, on GPU), `Script.runner_cache` (generation context)". Also `CLAUDE.md:25`:
"`seedvr2_neo/` — our helpers … (model discovery)".
The code now uses `devices.device` (`scripts/seedvr2_script.py:341-342`), `runner_cache` no longer exists (grep: only these
doc lines), and cached models sit in system RAM and only with "Keep in RAM". A reader of the architecture concludes
the device fix was not done (the block's acceptance criterion) and looks for a context cache that is gone.
Fix: "Forge's device (`modules.devices.device`); the vendored code still uses `cuda:0` in places (link to
`seedvr2-upstream.md`)"; caches row → "`GlobalModelCache` (models in RAM, only with Keep in RAM)"; list the three new
`seedvr2_neo` modules in `CLAUDE.md`.

### 2. Import of the pipeline catches only `ImportError` — 🟢 R2 (S2×L1) (open since fresh-eyes #8, not in `07-RISKS.md`)
`scripts/seedvr2_script.py:194-198`. The import runs `seedvr2_vendor/optimization/compatibility.py:697`
(`_probe_bfloat16_support`), which re-raises every `RuntimeError` except `CUBLAS_STATUS_NOT_SUPPORTED` (`:692-695`).
Scenario: a GPU/torch build mismatch ("no kernel image is available") or an OOM in the probe (since B06 the import
runs **before** the SD unload, i.e. with the checkpoint still in VRAM) → the exception leaves `run()` raw; Forge shows a
traceback toast instead of the `Error: …` result the step was built for. Still before the base image, so no generation
time is lost. Fix: catch `Exception` there (it is the boundary) with the same console line, or record it in `07-RISKS.md`.

### 3. Local-patch entry claims a live check is still pending; patch comment names the wrong caller — 🟢 R1 (S1×L1)
`docs/external-apis/seedvr2-upstream.md:35`: "to be checked in the live run" — the card's live check 1 records exactly
this ("DiT not cached - skipping CPU movement", no RAM jump at the end of Phase 2, VRAM returned after the phase).
`seedvr2_vendor/optimization/memory_manager.py:1054-1055`: "the caller collects garbage and empties the CUDA cache" —
the caller of `cleanup_dit` / `cleanup_vae` is `generation_phases.py:796` / `:1047`, which does neither; it is the Forge
script's `release_freed_memory()` after the phase (`scripts/seedvr2_script.py:381`, `:387`, `:393`). A future upstream sync reviewer
reading "the caller" will look in the wrong file. Fix: "checked live: …"; "the Forge script collects garbage and empties
the CUDA cache after the phase".

### 4. `05-DECISIONS.md` says the BlockSwap block count is "not measured yet" — 🟢 R1 (S1×L1)
`docs/05-DECISIONS.md:118-119`. It is measured (card live check 6, `07-RISKS.md` §3, `RECOMMENDED_7B_BLOCKS_TO_SWAP = 16`
with its measurement comment in `seedvr2_neo/memory_plan.py`). Fix: "depends on GPU, model and resolution (16 measured
for 7B fp8 at 1080px on 8 GB; fewer blocks fall off a cliff), so a fixed automatic value is wrong for other setups".

### 5. README promises unload after "an error" — a save error keeps the models — 🟢 R1 (S1×L1)
`README.md:56-57`: "After an error or an interrupt the models are always unloaded." `scripts/seedvr2_script.py:284`
sets `succeeded = True` before PIL conversion, `create_infotext` and `images.save_image`; `finish_run` then keeps the
models. Scenario: Keep in RAM, output folder not writable → UI shows `Runtime Error: …`, console "Models kept in RAM
for the next run", RAM stays 3–8 GB higher, contrary to the README. `09-PROCESSES.md` step 9 was already corrected;
the README was not. Fix: "After an error or an interrupt during the upscale the models are unloaded".

### 6. Acceptance criterion 1 has no recorded evidence — 🟢 R2 (S1×L2)
Card criterion: "A broken pipeline import (simulated) returns an error before the base image … — shown by a test or a
console log". `run()` has no tests (B09) and the card's live results (runs 1–8) contain no simulated import failure.
The code order is right (`scripts/seedvr2_script.py:195` import before `:222` `process_images` and `:247` SD unload),
but the criterion as written is unmet. Fix: one live run with `seedvr2_vendor` temporarily renamed, console line
recorded in the card — or reword the criterion to "by review" before ticking it.

### 7. `_upscale_tensor` and `run()` stay over the length guideline; `"cpu"` literal next to `CPU_DEVICE` — 🟢 R1 (S1×L1)
`scripts/seedvr2_script.py:334-399` (`_upscale_tensor`, ~65 lines: context + runner setup + four phases, two levels of
abstraction; open since code-audit #7) and `:183-331` (`run()`, ~150 lines, grew by ~20 in B06).
`scripts/seedvr2_script.py:345` passes `tensor_offload_device="cpu"` while `seedvr2_neo/memory_plan.py` defines
`CPU_DEVICE` for the same vendored contract. Fix: split `_upscale_tensor` into `_prepare_runner(...)` and
`_run_phases(...)`; use `CPU_DEVICE` (or a `tensor_offload_device` field in `MemoryPlan`). `run()` splitting can wait for
the B09 Forge stubs.

### 8. User-facing note: "on a 8.0 GB GPU" — 🟢 R1 (S1×L2)
`seedvr2_neo/memory_plan.py:110`. Every 8 GB user (the target class) sees "on a 8.0 GB GPU" (card live result 5 shows it).
Fix: the article depends on the number ("an 8", "a 10", "an 11"), so drop it: "7B model, GPU with {x:.1f} GB: …".

### 9. Patched branches keep a dead conditional — 🟢 R1 (S1×L1), optional
`seedvr2_vendor/optimization/memory_manager.py:1064` and `:1143`: `reason = "model caching" if cache_model else
"releasing GPU memory"` is now reached only with `cache_model=True`, so the `else` value is dead. Keeping upstream lines
untouched to ease the sync is a valid reason to leave it; if so, nothing to do. Otherwise `reason = "model caching"`.

---

## Checked and found correct (no finding)
- **Keep → Unload with the same files:** `prepare_run` keeps the cache (tracker valid), the vendored
  `_initialize_cache_context` removes both entries and the runner template because `dit_cache=False`
  (`model_configuration.py:609-612`, `:628-630`); `release_freed_memory()` after phase 1 collects the old copy before
  the DiT loads in phase 2; `finish_run` then forgets the tracker. No stale trust, no double DiT in RAM during phase 2.
- **Keep, other DiT or VAE:** tracker invalid → both entries and the template dropped before `prepare_runner`, so
  `_acquire_runner` builds a new runner; `_setup_dit_model`'s "model changed" branch (`cleanup_dit(cache_model=False)`,
  now affected by the patch) is unreachable in our flow because the template is gone.
- **The patch never drops a model that something else still holds:** with `cache_model=False` the model is never in
  `GlobalModelCache` (cached entries are set only when `dit_cache` / `vae_cache` is on, `generation_phases.py:310-323`,
  `:632-646`), so `runner.dit = None` / `runner.vae = None` releases the last strong reference. Keep mode goes through the
  unchanged `else` branch.
- **Error / interrupt in any phase:** phases 2 and 3 call `cleanup_dit` / `cleanup_vae` in `finally`; Keep mode then
  drops the cache entries in `finish_run`, Unload mode never cached them. The handled exception is released before the
  outer `finally`, so its `gc.collect()` reaches the frames' tensors.
- **`finish_run` in `finally` cannot mask the result:** `remove_dit` / `remove_vae` → `release_model_memory`, which
  catches `AttributeError` / `RuntimeError` itself.
- **Memory plan:** BlockSwap or I/O swap always sets a CPU offload device (the vendored `ValueError` cannot fire on CUDA);
  unknown "After run" → Unload; slider garbage, `inf`, `nan`, negatives clamp to 0..36 (tested).
- **7B note:** threshold boundary, unknown GPU, non-CUDA device, missing vendored package (lazy import with fallback) —
  all give no note or the right note; `.change` also fires when the 🔄 refresh changes the DiT value.
- **Tests really test:** lifecycle tests fail on keep-after-failure, ignoring Force Reload, ignoring a changed file,
  re-reading identities at the end, double release; identity tests fail on a name-only or path-only identity;
  `is_7b_checkpoint` tests use real safetensors headers, so a producer/consumer mismatch fails.

## For plan-drift-checker (not code findings)
`docs/04-STATE.md` still says `274 passed`; the run gives 329. The card's status line says "awaiting the owner's live
check (step 12)" while the results section records it, and no acceptance checkbox is ticked.
