# B06 code audit — VRAM/RAM lifecycle

Scope: `git diff dffc1a6..HEAD` (commit `54af56e`): `scripts/seedvr2_script.py`, `seedvr2_neo/memory_plan.py`,
`seedvr2_neo/loaded_models.py`, `seedvr2_neo/model_lifecycle.py`, `tests/seedvr2_neo/test_*.py`, docs.
Script calls checked against `seedvr2_vendor/` (generation_utils, generation_phases, model_configuration, model_cache,
memory_manager cleanup_dit/cleanup_vae/release_model_memory, blockswap validate/apply, interfaces/video_upscaler).

Tests: `324 passed, 1 skipped, 2 xfailed`.

Summary: 1 BLOCKER, 1 IMPORTANT, 5 MINOR.

---

## BLOCKER

### 1. The 7B low-VRAM note never appears: case mismatch "7B" vs "7b" — 🟠 R8 (S2×L4)
`seedvr2_neo/memory_plan.py:86` (and docstring `:83`); producer at `scripts/seedvr2_script.py:85`.

`describe_vram_warning` passes `choose_dit_size(...).size.value`. In `seedvr2_vendor/utils/dit_size.py:19-21`
`DitSize.B3 = "3B"`, `DitSize.B7 = "7B"` (upper case). `low_vram_warning` checks `if dit_size != "7b"` (lower case), so
for every real model it returns `None`. The docstring at `:83` claims the vendor returns `"3b" / "7b"` — it does not.

Proof (run in the repo):
```
DitSize.B7.value                                  -> '7B'
low_vram_warning(DitSize.B7.value, 8 * 1024**3)   -> None
```
Scenario: 8 GB GPU, user selects `seedvr2_ema_7b_fp8_e4m3fn.safetensors` → no note, no hint to turn on BlockSwap; the run
spills into shared memory (~13 min per image). Card step 9 and the acceptance criterion "With a 7B DiT selected on a GPU
below the threshold, the UI shows the warning" cannot pass; `09-PROCESSES.md` §1 step 5 describes behaviour that
does not happen.

Fix: compare against the vendor's value (e.g. `dit_size.lower() != "7b"`, or a module constant `"7B"` documented as
`DitSize.B7.value`), fix the docstring, and add a test fed with `DitSize.B7.value` (see finding 2).

---

## IMPORTANT

### 2. Tests of `low_vram_warning` check an invented contract, so they pass with the broken integration — 🟡 R6 (S2×L3)
`tests/seedvr2_neo/test_memory_plan.py:58-83`.

Every test passes the literal `"7b"`, which the real producer (`choose_dit_size(...).size.value`) never returns. The
`("3b", 8 * GIB)` "no warning" case passes for the wrong reason too ("3B" also fails the comparison). The suite is
green while the feature is dead (finding 1). Fix: build the input from `seedvr2_vendor.utils.dit_size.DitSize`
(stdlib-only module, importable in tests) or test `describe_vram_warning`'s pure part end-to-end with a fake 36-block
safetensors header, so a producer/consumer mismatch fails a test.

---

## MINOR

### 3. File identity is read after the run, not when the models were loaded — 🟢 R2 (S2×L1)
`seedvr2_neo/model_lifecycle.py:45-46` (`finish_run` → `tracker.remember`), called from `scripts/seedvr2_script.py:313`.

`remember()` stats the files at the end of the run. Scenario: "Keep in RAM", a 7B run takes minutes; meanwhile the user
copies a newer file with the same name over the DiT. The cache holds the old weights, the tracker records the new
file's size/mtime, and the next run reuses the old weights as "same file" — exactly what `loaded_models.py` exists to
prevent. Fix: read both identities in `prepare_run` (before loading) and pass them to `finish_run` to remember.

### 4. The "keep or unload" rule is duplicated in the script — 🟢 R1 (S1×L1)
`scripts/seedvr2_script.py:312-315` vs `seedvr2_neo/model_lifecycle.py:45`.

`keep_models = memory_plan.cache_models and succeeded` re-derives the decision `finish_run` makes internally, only to
choose the console line. If the rule in `finish_run` changes, the log lies ("Models kept in RAM" while they were
dropped). Fix: let `finish_run` return whether the models were kept and print from that.

### 5. 7B fp16 on a 12–16 GB GPU gets no note — 🟡 R4 (S2×L2)
`seedvr2_neo/memory_plan.py:19-21`, `:86-90`.

The threshold (11 GiB) is derived from 7B **fp8** weights (comment says so), but the check only looks at the size class,
not the precision. Scenario: 16 GB GPU (reports ~15.9 GiB ≥ 11), user picks `seedvr2_ema_7b_fp16.safetensors`
(~16.5 GB of weights) → no note, the DiT cannot fit and spills. Fix: either base the threshold on the file size of the
selected DiT plus a margin, or document the limitation in the note/`07-RISKS.md`. (Moot until finding 1 is fixed.)

### 6. `07-RISKS.md` understates when the DiT header is now read — 🟡 R4 (S4×L1)
`docs/07-RISKS.md:42`; code `scripts/seedvr2_script.py:106` (`ui()` → `describe_vram_warning(scan, scan.default_dit)`).

The risk line says a corrupt GGUF "hangs the UI on selection". The header of the **default** DiT is also read inside
`ui()`, i.e. while Forge builds both tabs at startup. Scenario: the default DiT is a GGUF with an absurd array length
(the existing R4 bug) → Forge hangs at startup, not just on a dropdown change; the user cannot reach the UI to pick
another model. Fix: say "at WebUI start (default DiT) and on selection" in the risk, or skip the check in `ui()` until
the bounded parser exists.

### 7. `_upscale_tensor`: 16 parameters, ~67 lines, unused `gen_info` — 🟢 R1 (S1×L1)
`scripts/seedvr2_script.py:317-384`, `:356`.

Over the code-style guideline (30–40 lines, one level of abstraction); the 16 positional parameters (several plain ints
in a row: seed, resolution, tile_size, tile_overlap) are easy to swap silently. `gen_info` is assigned and never used.
Fix: pass the per-run settings as one frozen dataclass; split context/runner setup from the phase sequence; use `_` for
the unused value.

---

## Checked and found correct (no finding)
- `run()` argument order matches the `ui()` return list; new controls are appended with defaults, so old API calls work.
- No `cuda:0` / `"0"` literal left; `devices.device` used for both DiT and VAE.
- Pipeline import happens before the base image and the SD unload.
- `prepare_runner` / `setup_generation_context` keyword names and types match the vendored signatures; the string
  `"cpu"` offload device is normalised by `setup_generation_context` and accepted by `validate_blockswap_config` /
  `apply_block_swap_to_dit` (only `.to()`/`str()` use).
- BlockSwap on → DiT offload device set, so the vendor's `ValueError` cannot fire; slider max 36 is clamped by the vendor
  for 3B.
- "Keep in RAM" without an offload device: `cleanup_dit` / `cleanup_vae` fall back to CPU, so cached models end in RAM.
- Error/interrupt in "Unload" mode: models are not in the global cache; the runner is only referenced from the exception's
  frames. Verified on Python 3.10 that after `return` inside `except`, `sys.exc_info()` is cleared and frame locals are
  freed before the outer `finally`, so `gc.collect()` + `torch_gc()` in `finish_run` releases them.
- Error/interrupt in "Keep in RAM" mode: `remove_dit` / `remove_vae` call `release_model_memory` and drop the
  `forge_dit+forge_vae` runner template.
- `find_model_file` prefers `os.path.join(scan.folder, name)` (local patch), the same path the tracker stats.
- `test_model_lifecycle.py` and `test_loaded_models.py` would fail on the obvious mutations (keep on failure, double
  release, missing `forget`, name-only identity).
