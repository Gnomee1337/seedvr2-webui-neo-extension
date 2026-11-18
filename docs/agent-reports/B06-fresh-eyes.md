# B06 — fresh-eyes review

Reviewed: `git diff dffc1a6..HEAD` (commits `54af56e`, `3afc913`): `scripts/seedvr2_script.py`, `seedvr2_neo/memory_plan.py`,
`seedvr2_neo/loaded_models.py`, `seedvr2_neo/model_lifecycle.py`, tests in `tests/seedvr2_neo/`, the docs changed with them.
Checked against the vendored code: `core/generation_utils.py`, `core/generation_phases.py`, `core/model_configuration.py`,
`core/model_cache.py`, `optimization/memory_manager.py`, `optimization/blockswap.py`, `optimization/compatibility.py`,
`interfaces/video_upscaler.py` (how upstream's own node calls the same functions).
The findings code-auditor already raised and `3afc913` fixed (enum comparison of the 7B check, identities read in
`prepare_run`, `finish_run` return value, `UpscaleRequest`) are not repeated.
`tests/seedvr2_neo`: 125 passed, 1 skipped.

## How I understood the result

Before any work `run()` checks the model selection and imports the vendored pipeline (an `ImportError` returns an empty
result before the base image and the SD unload). The UI gets "After run" (Unload SeedVR2 by default / Keep in RAM), a
BlockSwap slider and an "I/O layers" checkbox; BlockSwap or I/O swap set CPU as the offload device of both DiT and VAE.
Each run builds a new generation context on `modules.devices.device`, asks the module-level `ModelLifecycle` to drop the
vendored cache unless the cached models came from the same files (path, size, mtime, read before loading) and Force
Reload is off, runs the four phases with `devices.torch_gc()` after the first three, and at the end keeps the models
only after a successful "Keep in RAM" run, otherwise removes them, then `gc.collect()` + `torch_gc()`. A note under the
DiT dropdown warns about a 7B checkpoint on a GPU below 11 GiB.
This matches the task. Nothing outside the task was added, apart from two `06-IDEAS.md` entries.

## Silently changed behaviour (old vs new `scripts/seedvr2_script.py`)

Checked, **unchanged**: the saved file name and seed (`args.seed` became `actual_seed`, which is already an `int` in all
three branches), the `Processed` return values (success, interrupt, error), the infotext keys, the `-original` copy,
the txt2img self-removal from `p.scripts.scripts`, the order of the seed resolution and the SD unload.
Changed, and stated in the task or the docs: the pipeline import moved before the base image; Force Reload drops the models;
"Unload" (the default) reloads both models from disk every run; the context is no longer kept on `self`; text embeddings
are loaded inside phase 2 by the vendored code instead of explicitly by the script (same result).
Changed and **not mentioned anywhere**: see findings 5 (console messages) and 9 (when the vendored import side effects happen).

## Findings

No BLOCKER.

### IMPORTANT

**1. With "Unload SeedVR2" (the default) each model is still copied whole from the GPU to RAM before it is deleted.**
`seedvr2_vendor/optimization/memory_manager.py:1052-1063` (DiT) and `:1130-1141` (VAE): with `cache_model=False` and no
offload device, `cleanup_dit` / `cleanup_vae` move the model to `cpu` ("releasing GPU memory") and only then call
`release_model_memory`, which frees only CUDA tensors. The script passes `cache_model=False` in Unload mode
(`scripts/seedvr2_script.py:382`, `:388`). So at the end of phase 2 the whole DiT (7B fp8 ≈ 8 GB, 3B ≈ 3–6 GB) appears in
system RAM for a moment, plus the PCIe copy time; the DiT is freed by reference counting afterwards. The RAM after the run
is fine (the acceptance criterion measures only that); the peak is not. On the target class (16 GB RAM, SD checkpoint just
offloaded to RAM by "Unload SD Checkpoint") this is the same starvation that `07-RISKS.md` described for caching, only
brief. Not documented: `09-PROCESSES.md` process 5 step 9 says the pipeline moves models off the GPU "after their phase"
only for "Keep in RAM".
Fix options: a vendored patch that skips the CPU move when not caching (as it already does on MPS), or at least a line in
`07-RISKS.md` and a "watch the peak RAM at the end of Phase 2" step in the live check.
R = S2 × L3 = 6 🟡

### MINOR

**2. "0 = off" is not true when "I/O layers too" is ticked.** `scripts/seedvr2_script.py:148-150`,
`seedvr2_neo/memory_plan.py:64`. Slider at 0 + checkbox on → BlockSwap is active (I/O swap) **and** the VAE is offloaded to
CPU between phases. A user reads the slider label and believes BlockSwap is off. Either name the checkbox as part of
BlockSwap ("also swap I/O layers; turns BlockSwap on") or ignore the checkbox at 0 blocks.
R = S1 × L2 = 2 🟢

**3. The 7B note is keyed on the model family, not on the checkpoint size.** `seedvr2_neo/memory_plan.py:83-97`. The
false negative (7B fp16 on 12–16 GB) is already in `07-RISKS.md`; the false positive is not: a small 7B GGUF (Q4, about
half of fp8) on an 8 GB card gets "spills into shared memory and runs many times slower". The block card step 3 said
"model size × GPU total memory". The docstring "a wrong warning is worse than none" contradicts this case.
R = S1 × L2 = 2 🟢

**4. A changed VAE also drops the cached DiT (and vice versa).** `seedvr2_neo/loaded_models.py:40-44`,
`seedvr2_neo/model_lifecycle.py:42-44`. Validity is all-or-nothing for the pair, so switching only the VAE in Keep mode
reloads a 7B DiT from disk (minutes on slow drives), although the vendored cache can keep the two independently.
R = S1 × L2 = 2 🟢

**5. Console messages: what the user is told changed.** `scripts/seedvr2_script.py:269-270`, `:328-330`.
- "Cache cleared." now also appears (debug only) when the cache was dropped because a file changed, not only for Force
  Reload; without debug the user is never told why the next run reloaded the models.
- A new line is always printed at the end ("Models kept…" / "Models unloaded, memory released."), even after errors that
  happened before anything was loaded.
Harmless, but a reader of a log will misread "Cache cleared." as "Force Reload was on".
R = S1 × L2 = 2 🟢

**6. `09-PROCESSES.md:90` says models are unloaded after any exception; one exception keeps them.** `succeeded = True` is
set at `scripts/seedvr2_script.py:283`, before PIL conversion, `create_infotext` and `images.save_image`. A save failure
(disk full, no permission) gives `Runtime Error: …` in the UI while the console says "Models kept in RAM for the next run."
The behaviour is defensible (the models are intact), the doc is not exact.
R = S1 × L1 = 1 🟢

**7. The startup note is computed for `scan.default_dit`, not for the value Forge restores.** `scripts/seedvr2_script.py:120`.
Forge applies saved values from `ui-config.json` to the dropdown after `ui()` built it; no `.change` event fires for that,
so the note can show for a 3B or stay hidden for a 7B until the user touches the dropdown. Not verified in Forge — check
in the live run by saving a 7B as the default.
R = S1 × L2 = 2 🟢

**8. `import_pipeline` docstring no longer holds.** `scripts/seedvr2_script.py:56-57`: "not at module level, so the UI
still loads" — the script now imports `seedvr2_vendor.utils.dit_size` at module level through
`seedvr2_neo/memory_plan.py:14`, so `seedvr2_neo` depends on the vendored package. The module is light (stdlib only,
`gguf` lazily), so nothing breaks today; the comment and the "seedvr2_neo has no Forge/torch imports" note in
`02-ARCHITECTURE.md` mislead the next reader about which import can make the script vanish. Also "Done first in run()" —
the model selection check is first. Only `ImportError` is caught (`:195`); the vendored import can also raise
`RuntimeError` (bf16 probe, `seedvr2_vendor/optimization/compatibility.py:686-695` re-raises any non-CUBLAS error) or
`OSError` (DLL load), which still escape `run()` as a raw Forge error — now before the base image, which is the intent,
but without the friendly `Error: …`.
R = S1 × L2 = 2 🟢

**9. Vendored import side effects now happen before the first base image.** `seedvr2_vendor/optimization/compatibility.py:111-116`
installs stub modules into `sys.modules` when `flash_attn`, `xformers._C_flashattention`, `bitsandbytes` or
`triton.ops.matmul_perf_model` fail to import, and `:686-697` allocates a test tensor on hard-coded `cuda:0`. Before B06
this ran after the first base image; now it runs before it. Process-wide it already affected every generation after the
first SeedVR2 run, so the change is small, but these shims are documented nowhere (`07-RISKS.md`, `seedvr2-upstream.md`):
a Forge feature that probes `bitsandbytes` / `flash_attn` after our import sees a stub.
R = S2 × L1 = 2 🟢

**10. "Forge's device" covers only what the script passes.** The vendored code still uses `cuda:LOCAL_RANK`/`cuda:0` on its
own: `seedvr2_vendor/common/distributed/basic.py:47-53` (`get_device`, used by `core/infer.py:90,128,213`,
`optimization/blockswap.py:209`), `optimization/compatibility.py:688`, `optimization/memory_manager.py:111,157,417`.
It works because `--gpu-device-id` hides other GPUs (index 0 = chosen GPU, `forge-neo.md`). Two consequences nobody wrote
down: XPU/other backends still fail; and on a CPU-only Forge (`devices.device` = cpu) turning BlockSwap on raises
`ValueError … Set --dit_offload_device cpu` (`optimization/blockswap.py:99-117`), naming a flag Forge does not have.
`02-ARCHITECTURE.md` "running the four phases on Forge's device" reads as stronger than it is.
R = S1 × L1 = 1 🟢

**11. Between phases only the allocator cache is released, not unreachable Python objects.**
`scripts/seedvr2_script.py:379,385,391` call `devices.torch_gc()` without `gc.collect()`. A model the vendored code drops
inside a phase and that sits in a reference cycle (BlockSwap wraps `forward` and `to` with closures that reference the
model — `optimization/blockswap.py:255-278`) keeps its memory until `finish_run`. Same for "Keep in RAM" → "Unload" with the
same files: `prepare_run` keeps the cache (files valid), then the vendored `_initialize_cache_context`
(`core/model_configuration.py:609-612`) removes it without `gc`, while a fresh copy is loaded — two copies in RAM if the
old one is cyclic. Not measured; `release_freed_memory` (`gc` + `torch_gc`) at the phase boundaries would remove the doubt.
R = S2 × L1 = 2 🟢

**12. After Forge's "Reload UI" kept models stay in RAM with nobody tracking them.** The script module is re-executed, so
`MODEL_LIFECYCLE` (`scripts/seedvr2_script.py:52`) starts with an empty tracker, while the vendored `GlobalModelCache` lives
in `sys.modules` and still holds the models. The next run drops them (safe direction), but until then 3–8 GB stay in RAM.
R = S2 × L1 = 2 🟢

**13. Docstring promises an unverified result.** `scripts/seedvr2_script.py:336-337`: "so the next phase does not spill
into shared memory" — `07-RISKS.md` §3 says exactly this awaits the live check. A reader takes it as established.
R = S1 × L1 = 1 🟢

## Hidden assumptions (no finding, worth keeping in mind)

- `ModelLifecycle` holds one pending run (`seedvr2_neo/model_lifecycle.py:27-33`, `_run_files`): correct only while Forge
  serialises generations (UI and API both go through the queue lock). Two concurrent runs would overwrite each other's file
  identities.
- Strings as offload devices (`"cpu"`) are accepted everywhere in the vendored code because it compares devices through
  `str()` (`optimization/blockswap.py:99-101,759`) or `torch.device(...)`; upstream's own node passes `torch.device`. A
  future upstream sync that compares `==` against `torch.device` would break this silently.
- `AfterRun` values are the UI labels (`seedvr2_neo/memory_plan.py:29-33`): renaming a label turns saved/API values into
  "unknown" → Unload. Safe, but a label edit changes behaviour.
- The vendored `release_tensor_memory(ctx['input_images'])` at the end of phase 4
  (`core/generation_phases.py:1471-1473`) empties the script's `img_tensor` in place; nothing reads it afterwards today.

## Not found

- Cache trusted wrongly: no path found where cached weights from another file or a failed run are reused. Unload mode
  cannot reuse the cache at all (the vendored code removes it when `dit_cache=False`), and errors/interrupts drop it.
- Memory still referenced after a run: the context, runner and text embeddings are locals of `_upscale_tensor`; the
  exception (and its frames) is released before the `finally` runs, so `gc.collect()` there can reach them. RoPE
  `lru_cache` is cleared by `cleanup_dit` in phase 2's `finally`.
- BlockSwap configurations rejected by the vendored code: none on CUDA (offload device always set when BlockSwap is on;
  block count clamped by both sides). Only the CPU-only case of finding 10.
- txt2img vs img2img instances: they share the cache ids and the lifecycle by design; a model kept by one tab and
  requested by the other with the same files is reused correctly, other files drop it.

## Re-check after `5f1bac0` and `65beae4`

Reviewed `git diff 3afc913..HEAD`.

### Finding 1 (Unload copied the model to RAM): fixed. The stated reason is wrong; the result holds for a different reason

**The patch** (`seedvr2_vendor/optimization/memory_manager.py:1052-1059`, `:1132-1138`) changes only the condition
`param_device.type == 'mps' and not cache_model` → `not cache_model`.
- **Keep in RAM (`cache_model=True`)**: both the old and the new condition are false, so the `else` branch (move to the
  offload device or CPU) runs exactly as before. Unchanged.
- **Unload, no BlockSwap**: the CPU move is skipped, then `release_model_memory`, `runner.dit = None`.
- **Unload, BlockSwap without I/O swap**: the first parameter is an I/O layer on the GPU. Before the patch, the move went
  through `_handle_blockswap_model_movement` and copied every block to the CPU. Now it is skipped.
  `cleanup_blockswap(keep_state_for_cache=False)` (`optimization/blockswap.py:860-935`) restores the wrapped forwards,
  moves only the non-block children (I/O layers, small) to the offload device and drops the BlockSwap attributes.
  The wrappers hold only `weakref`s to the model and runner (`blockswap.py:481-482,560-561,733`), so no strong
  reference survives. Correct.
- **Unload, BlockSwap with I/O swap**: the first parameter is on the CPU, so the `param_device.type not in ['meta', 'cpu']`
  branch is not entered at all, the same as before the patch. The blocks still on the GPU are never copied. Correct.

**Why the comment is wrong.** The patch comment (`memory_manager.py:1054-1056`) and the `seedvr2-upstream.md` entry
(lines 29-32) say "`release_model_memory` frees the GPU storage in place". It does not. `param.data.set_()`
(`memory_manager.py:564,571`) acts on the detached tensor that `.data` returns, not on the parameter. I checked this on
torch 2.15 (CPU): after `param.data.set_()` the parameter still holds its full storage (4 194 304 bytes before and after
for a 1024×1024 `Linear`). The tensor mechanics are the same on CUDA. In this run the GPU memory is freed because
`runner.dit` / `runner.vae` is set to `None`. That drops the last strong reference, and then `release_freed_memory()`
(now `gc.collect()` + `torch_gc()` after every phase) returns the memory. This works today: no global or cache reference
to the model remains in Unload mode (I found no strong reference in `core/`, `optimization/` or `models/`).

The release now depends entirely on the model being unreachable. Before the patch, the CPU move freed the VRAM even if
something still held the model. If a future upstream sync or one of our changes keeps a reference (a debug helper, a
cache entry, a stored exception), the VRAM stays occupied silently instead of being moved out. R = S3 × L1 = 3 🟢.
- Fix the comment and the `seedvr2-upstream.md` entry: the memory is freed by dropping the reference plus `gc`, not in place.
- Optionally replace `param.data.set_()` with `param.set_()` under `torch.no_grad()`, or `param.data = torch.empty(0, device=...)`, so it really frees in place.

**Wrong verification claim.** `seedvr2-upstream.md:32` says "Not unit-tested (tests run without torch); checked live."
The block card shows step 12 (live check) as not done yet, and live check step 1 was only just extended to watch for
the RAM peak. Until the owner runs it, "checked live" is not true. Write "to be checked in the live run" until then.

### MINOR findings

| # | Finding | Status |
|---|---|---|
| 2 | "0 = off" while I/O swap is on | **Fixed in the UI**: the slider label no longer says "0 = off" (`scripts/seedvr2_script.py:148`). **Still open in the docs**: `docs/09-PROCESSES.md:36` still says "BlockSwap blocks (0–36, 0 = off)", and the card (line 28) says the same. The checkbox alone still turns BlockSwap on; nothing in the UI says so. |
| 3 | 7B note keyed on family, not size | **Moved to `07-RISKS.md`** (R4, "judges the model family, not its size") |
| 4 | VAE change drops the DiT | **Moved to `07-RISKS.md`** (R2, "reloads both models when only one changes") |
| 5 | Console messages | **Fixed**: the message is always shown now (`scripts/seedvr2_script.py:269-270`). Its text gives the wrong reasons, though: "(Force Reload, other files or a previous failed run)". A failed run already empties the cache in `finish_run`, so the next `prepare_run` finds nothing to drop and prints nothing. The real third case is a fresh tracker after Forge's "Reload UI" or a script reload. 🟢 R1, still open |
| 6 | Save failure keeps models, doc said otherwise | **Fixed** in `09-PROCESSES.md` (steps 9 and "On failure") |
| 7 | Startup note for the default DiT, not the restored one | **Moved to `07-RISKS.md`** (inside the 7B-note entry, marked "not checked in Forge") |
| 8 | Module-level vendored import / docstring | **Import fixed**: `dit_size` is now imported lazily in `is_7b_checkpoint` (`seedvr2_neo/memory_plan.py:81-92`), with an `ImportError` fallback. **Still open (wording)**: the `import_pipeline` docstring (`scripts/seedvr2_script.py:56`) still says "Done first in run()", but the model check runs first. Only `ImportError` is caught at `:195`; a `RuntimeError`/`OSError` at import still escapes raw. 🟢 R1 |
| 9 | Vendored import side effects | **Moved to `external-apis/seedvr2-upstream.md`** ("Import side effects") |
| 10 | Vendored code picks devices itself; CPU-only BlockSwap error | **Moved to `external-apis/seedvr2-upstream.md`**. `02-ARCHITECTURE.md:66` still says "running the four phases on Forge's device" without the caveat. Acceptable, since the caveat is one link away |
| 11 | No `gc.collect()` between phases | **Fixed**: `release_freed_memory()` after phases 1–3 (`scripts/seedvr2_script.py:380,386,392`). The new patch now depends on this (see above) |
| 12 | Kept models untracked after "Reload UI" | **Moved to `07-RISKS.md`** (in the R2 entry with finding 4) |
| 13 | Docstring promised no spill | **Fixed** (`scripts/seedvr2_script.py:336-338`, "less likely to spill") |

### Summary of the re-check
No BLOCKER and no IMPORTANT left. Still open, all MINOR:
1. The patch comment and the `seedvr2-upstream.md` entry give the wrong mechanism ("in place"), and the entry claims a live check that has not happened. R3.
2. The "dropped" message lists "a previous failed run", which cannot trigger it. R1.
3. `09-PROCESSES.md:36` still says "0 = off". R1.
4. The `import_pipeline` docstring still says "Done first". R1.
