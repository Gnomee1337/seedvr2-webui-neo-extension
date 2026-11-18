# B06 final fresh-eyes review

Reviewed: `git diff main...HEAD` (7 commits, `54af56e`..`30775da`), with a read of the vendored code it calls
(`core/model_configuration.py`, `core/model_cache.py`, `core/generation_phases.py`, `optimization/memory_manager.py`,
`optimization/blockswap.py`). Tests: 329 passed, 1 skipped, 2 xfailed (`uv run … pytest tests -q`).
Ratings: `docs/07-RISKS.md` (R = S × L).

## How I understood the result

On every run the script now builds a "memory plan" from three new controls (After run: Unload / Keep in RAM;
BlockSwap block count; BlockSwap I/O layers). Before loading, a process-wide `ModelLifecycle` drops the vendored model
cache unless the selected DiT and VAE files (absolute path, size, mtime) are the ones it was filled from, or always on Force
Reload. The pipeline gets Forge's device, a new generation context every run, and `gc` + empty-cache after phases 1–3.
After the run the models stay cached only after a successful "Keep in RAM" run; otherwise both are removed and memory released.
A vendored patch drops non-cached models straight from the GPU instead of copying them to RAM first. A 7B DiT on a GPU
under 11 GiB gets a UI note recommending 16 BlockSwap blocks; the README has tips for 8–10 GB GPUs.
This matches the task. I found nothing extra beyond what was asked.

## Silently changed behaviour (old vs new `scripts/seedvr2_script.py`)

- Text embeddings: the script used to load them into the cached context once; now the vendored phase 2 loads them on
  every run (`generation_phases.py:593`). The same function and file, just more often. Fine.
- Saved image seed and `Processed` seed: `args.seed` → `actual_seed`, the same int. No change.
- Unload mode after a previous Keep run with the same files: the tracker trusts the cache, but the vendored
  `_initialize_cache_context` (`model_configuration.py:609-612`, `:627-629`) removes it itself because `dit_cache=False`.
  The cache ends up dropped, which is correct. Nothing reuses stale weights.
- All other differences are stated in the task (import first, device, unload by default, Force Reload, fresh context).

## Findings

No BLOCKER. No IMPORTANT.

### MINOR

**1. The "3–8 GB of RAM" cost of Keep in RAM is too low for fp16 models.** `README.md:56`, also `docs/05-DECISIONS.md:106`,
`docs/07-RISKS.md:74`. 7B fp16 is ~16 GB of weights and 3B fp16 + VAE is ~7 GB. A user with 16 GB RAM and a 7B fp16 (12–24 GB
GPU owners) picks Keep in RAM based on the README and the system starts swapping after the run. Write the cost as
"about the size of the DiT file plus the VAE".
R = S2 × L2 = 4 🟡

**2. `07-RISKS.md:75` says "the option's hint says what it costs". It does not.** `scripts/seedvr2_script.py:148`: the hint
is "Unload frees VRAM and RAM; Keep in RAM makes the next run start faster". RAM is not mentioned, so a reader of the
risk list thinks the UI covers the risk. Either add "(holds the models in system RAM)" to the hint or fix the risk text.
R = S1 × L2 = 2 🟢

**3. README: "After an error or an interrupt the models are always unloaded" is broader than the code.** `README.md:56-57`.
Models are dropped only in the `finally` around the pipeline (`scripts/seedvr2_script.py:327-331`). Suppose the interrupt
or error happens earlier: during the txt2img base image (`:228-230`), in img2img before the start (`:240-241`), or in the
SD unload. Then `run()` returns or raises before `prepare_run`/`finish_run`, and the models kept by the previous run stay
in RAM. This is the safe direction, but the sentence promises more than the code does. Add "during the upscale".
R = S1 × L2 = 2 🟢

**4. The user-visible warning says "on a 8.0 GB GPU".** `seedvr2_neo/memory_plan.py:110`. The article is wrong for 8 and 11
("an 8.0 GB"), and the live-check log in the card shows users see exactly this. Rephrase without the article: "7B model,
GPU has 8.0 GB: …".
R = S1 × L3 = 3 🟢

**5. The README heading and the warning threshold disagree.** `README.md:59` "GPUs with 8–10 GB" vs
`seedvr2_neo/memory_plan.py:21-22`: the note fires below 11 GiB and its comment names "8, 10 and 11 GB cards". An
11 GB card owner gets the note in the UI, but the README section title says it is not for them. Use "8–11 GB" or
"under 12 GB".
R = S1 × L2 = 2 🟢

**6. Force Reload does nothing in the default mode, and nothing says so.** `scripts/seedvr2_script.py:142`, `README.md:57`.
With "Unload SeedVR2" nothing is cached, so every run already reads from disk and the checkbox is a no-op. The README
sentence "Force Reload reads the model files from disk again" reads as a separate useful action. Add "(only matters with
Keep in RAM)" to the README and/or an `info=` on the checkbox.
R = S1 × L2 = 2 🟢

**7. Stale doc after the live check.** `docs/external-apis/seedvr2-upstream.md:35`: the local-patch entry still says
"to be checked in the live run". Live check 1 in the B06 card passed ("DiT not cached - skipping CPU movement", no RAM
growth). `docs/05-DECISIONS.md:118-119`: "the right block count … is not measured yet". It has now been measured for 7B fp8 at
1080px on 8 GB, and the note recommends 16. Both make a reader think the patch and the recommendation are unverified.
R = S1 × L2 = 2 🟢

**8. Dead expression left inside the vendored patch.** `seedvr2_vendor/optimization/memory_manager.py:1064` and `:1143`:
`reason = "model caching" if cache_model else "releasing GPU memory"` now sits in a branch reached only with
`cache_model=True`, so the `else` part can never be chosen. A reader of the patch looks for the path where the move
"releases GPU memory" and finds none. Keeping the upstream line makes the next upstream sync easier, so leave it, but
mention it in the patch comment or in `seedvr2-upstream.md`.
R = S1 × L1 = 1 🟢

**9. `output_tensor = None` in `finally` has no explanation.** `scripts/seedvr2_script.py:328`. On first reading it
looks like a leftover. It drops the last script-side reference to the pipeline's output before `finish_run` collects
garbage and empties the CUDA cache. One short comment would say so. Also note `_upscale_tensor`
(`:334`) takes both `debug` (the vendored Debug object) and `debug_mode` (bool), two near-identical names with different
types. `log_phases` / `show_phase_lines` would read better.
R = S1 × L2 = 2 🟢

**10. "BlockSwap: I/O layers too" alone turns BlockSwap on, and the UI does not say it.** `scripts/seedvr2_script.py:151`,
`seedvr2_neo/memory_plan.py:75`. With the slider at 0 and the box ticked, the DiT I/O layers swap and the VAE is moved
to RAM between phases (slower). `09-PROCESSES.md` documents it, the UI does not. This was carried over from the earlier
review and is still open in the UI. Add an `info=` such as "also swaps the DiT input/output layers; turns BlockSwap on".
R = S1 × L2 = 2 🟢

**11. Acceptance criterion 1 has no recorded evidence.** `docs/roadmap-blocks/B06-vram-ram-lifecycle.md:39`: "A broken
pipeline import (simulated) returns an error before the base image … shown by a test or a console log". `run()` has no
tests (B09), and the live-check results (`:96-110`) do not include a simulated import failure. The order is right when
I read the code (`scripts/seedvr2_script.py:194-198` comes before `:209`), but the criterion asks for a demonstration.
Either record a quick live check, or state in the card that code review closes it.
R = S1 × L2 = 2 🟢

## Hidden assumptions (no finding)

- File identity is (normalised absolute path, size, `mtime_ns`) (`seedvr2_neo/loaded_models.py:23-30`). A file replaced
  by a copy that keeps both size and mtime (`cp -p`, `robocopy` with timestamps) of a different file of the same size would be trusted.
  This is practically impossible for model files, but it is the limit of the check.
- `finish_run` runs inside `finally` (`scripts/seedvr2_script.py:329`). If it raises (e.g. the vendored `release_model_memory`
  fails), that exception replaces the already-built `Processed`. The image is saved on disk, but the UI shows a Forge error.
- `release_freed_memory` frees VRAM only if nothing else references the models. That holds today (checked in the previous
  review and live check 1) and is recorded in `07-RISKS.md` §3 (R3).
- `MODEL_LIFECYCLE` keeps one pending run. This is correct while Forge serialises UI and API generations.

## Not found

- No path where a cached model from different files, or after a failed or interrupted pipeline run, is reused.
- No path where Unload mode leaves the DiT/VAE referenced after the run. The context, runner and exception frames are gone
  before `finish_run` calls `gc.collect()`.
- The BlockSwap config is always valid on CUDA: the offload device is set whenever either control is on, and the block
  count is clamped.
