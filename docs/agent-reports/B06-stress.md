# B06 stress test — memory_plan, loaded_models, model_lifecycle

Scope: `seedvr2_neo/memory_plan.py`, `seedvr2_neo/loaded_models.py`, `seedvr2_neo/model_lifecycle.py`. Application code
not changed; only tests added.

Run: `uv run --no-project --python 3.10 --with-requirements requirements-dev.txt python -m pytest tests -q`
→ **994 passed, 1 skipped (POSIX-only, existing), 12 xfailed** (2 existing + 10 new strict xfails = 4 findings).

## New test files

| File | Result | What it covers |
|---|---|---|
| `tests/seedvr2_neo/test_memory_plan_stress.py` | 604 passed, 8 xfailed | "After run" of every type (None, case/whitespace variants, NUL, emoji, 110 KB string, list/dict/bytes/bool, enum members); BlockSwap slider of every type (bool, 10^1000, ±1e308, -inf, Decimal, Fraction, Unicode digits, "8.0", "1e3", 5000-digit string, bytes, complex, containers); I/O swap checkbox values; plan invariants over a 455-combination grid (offload device set iff BlockSwap on, VAE device = DiT device, exact `int`/`bool` types); frozen plan and fresh `block_swap_config` dict; low-VRAM note at real card sizes (8/10/11/12 GB) and extremes; `is_7b_checkpoint` with 10 corrupt safetensors headers × 3 names, unknown block counts, a directory in place of the file, Unicode/emoji folders, a >260-char path, garbage paths (`""`, NUL, `::::`, `\\?\`), a `.gguf` name without the `gguf` package, the vendored module missing / shadowed / broken |
| `tests/seedvr2_neo/test_loaded_models_stress.py` | 28 passed, 1 xfailed | Unicode, emoji and >260-char paths; `..` and doubled separators; Windows path case; symlink (not trusted — safe); bytes vs str path; directory in place of the model; `os.stat` raising PermissionError / OSError / FileNotFoundError; `""`, `.`, separator; NUL in path; file replaced by `shutil.copy2` (foreign mtime), by `os.replace` with the same size, deleted and recreated; swapped DiT/VAE; partial `remember` with `None`; last `remember` wins; every identity field compared; 200 remembers |
| `tests/seedvr2_neo/test_model_lifecycle_stress.py` | 33 passed, 1 xfailed | Double `finish_run`, `finish_run` without `prepare_run`, repeated `prepare_run` without finish (different and same files), Force Reload on an empty cache, falsy non-bool flags; six kinds of file change between prepare and finish (rewrite, touch, delete, delete+recreate for DiT/VAE); file missing at prepare; same file through `..`; swapped paths; Unicode/long folders; `""`/`.`/missing paths; `remove_dit`/`remove_vae` raising in prepare, finish and `unload`; `release_memory` raising in prepare, unloading finish and keeping finish; 500 mixed runs over 5 model pairs checking every drop, keep and release count against a model of the rules |

## Findings

Sorted by risk. None is a BLOCKER: the core guarantees hold (changed files are never trusted, a failure never leaves the
tracker trusting the cache, exactly one release per finish, BlockSwap never on without an offload device).

### 1. Any non-ImportError failure of the size check escapes `is_7b_checkpoint` — 🟢 R3 (S3×L1)
`seedvr2_neo/memory_plan.py:91-96`. Test: `test_any_failure_of_the_size_check_means_no_note` (3 params, strict xfail).

Input → wrong result: `seedvr2_vendor.utils.dit_size` import raises `SyntaxError` (damaged file after an interrupted
update) or `OSError`, or a foreign `dit_size` with another `choose_dit_size` signature raises `TypeError` → the exception
leaves `is_7b_checkpoint`. It is called from `Script.ui()` (`scripts/seedvr2_script.py:121`, via
`describe_vram_warning`), so Forge drops the whole script UI — exactly what the docstring says must not happen. Only
`ImportError` is caught. Expected: `False` (no note) for any failure of an optional UI hint.

### 2. `unload()` stops at a raising `remove_dit`: VAE stays cached, memory not released — 🟢 R2 (S2×L1)
`seedvr2_neo/model_lifecycle.py:63-69`. Test: `test_failing_dit_remove_still_removes_the_vae_and_releases` (strict xfail).

Input → wrong result: `cache.remove_dit` raises → `remove_vae` and `release_memory` never run; the VAE (hundreds of MB)
stays in the vendored cache and VRAM/RAM is held until the next run's prepare. The tracker is forgotten first, so
nothing stale is reused (tested, passes). Likelihood is low: the vendored `remove_dit` → `release_model_memory`
(`seedvr2_vendor/optimization/memory_manager.py:579`) swallows `RuntimeError`/`AttributeError`.

### 3. String checkbox values: `bool("False")` turns the I/O swap on — 🟢 R1 (S1×L1)
`seedvr2_neo/memory_plan.py:65`. Test: `test_swap_io_false_strings_mean_off` (5 params, strict xfail).

Input → wrong result: `swap_io_components="False"` / `"false"` / `"0"` / `"off"` / `"no"` (API `script_args` written
by hand) → `swap_io_components=True`, DiT and VAE offloaded to CPU: slower run, still correct. Inconsistent with
`clamp_blocks_to_swap`, which does parse `"8"`. The Gradio checkbox and `ui-config.json` always give a real bool.

### 4. A NUL in a model path raises instead of returning None — 🟢 R1 (S1×L1)
`seedvr2_neo/loaded_models.py:25-29`. Test: `test_path_with_nul_has_no_identity` (strict xfail).

Input → wrong result: `read_model_file_identity("...\\bad\0name.safetensors")` → `ValueError: embedded null character`
(from `os.path.abspath`, outside the `try`, and from `os.stat`, which raises `ValueError`, not `OSError`). The docstring
promises `None` for any unreadable path. Not reachable today: `run()` validates names against the scan before joining.

## Observations (tested behaviour is acceptable; no xfail)

- **mtime granularity.** NTFS stores mtime in 100 ns steps (a `+1 ns` change was lost in a first test draft), exFAT in
  10 ms, FAT32 in 2 s. A same-size replacement within one step, or a different file of the same size copied with an
  identical mtime, is trusted. Practically unreachable for multi-GB model downloads; adding `st_ino` to the identity
  would also catch `os.replace` swaps on NTFS/ext4.
- **`""` and directories have an identity.** `read_model_file_identity("")` is the working directory's identity, and a
  directory named like a model is tracked like a file (changes inside it are caught). Unreachable from the script; the
  loader would fail and the run unload anyway.
- **`clamp_blocks_to_swap("8.0")` / `("1e3")` is 0** while `"8"` is 8 and `8.0` is 8: BlockSwap silently off for a
  string float from the API. An `__int__` raising anything other than TypeError/ValueError/OverflowError (e.g. a
  multi-element torch tensor's `RuntimeError`) escapes. Neither reachable from Gradio.
- **`release_memory` raising propagates out of `prepare_run`/`finish_run`.** State stays safe in every path (cache
  emptied, tracker forgotten; keep path consumes the pending run — all tested). In the script, `finish_run` runs in a
  `finally` (`scripts/seedvr2_script.py:329`), so a CUDA error from `torch_gc` replaces the run's "Runtime Error"
  result with a raw exception. Script-level choice, not covered here.
- **Note text just below the threshold** reads "on a 11.0 GB GPU" (10.99… GiB rounds up). Cosmetic.
- **Symlinked model path** is a different identity → cache dropped, models reloaded. Safe side, costs one reload.
- Long paths (>260 chars) work in this environment for both identity and header reads; on systems without long-path
  support identity is `None` → never trusted (safe), and the 7B check falls back to the file name.
