# B06 — VRAM and RAM lifecycle, low-VRAM path for 7B

**Status:** 🔄 code and tests done (steps 1–11), awaiting the owner's live check (step 12)
**Weight:** 4 (from `docs/03-ROADMAP.md`)
**Decisions this block rests on:** "Unload the SD checkpoint before SeedVR2 by default"; "SeedVR2 is unloaded after
every run by default"; "Models are dropped after an error or an interrupt"; "The cached models are trusted only for the
same files"; "BlockSwap is a manual slider with a note for 7B on small GPUs"; "Fix in `scripts/` first".

## Why this block
After a run SeedVR2 keeps 3–8 GB in system RAM, reserved VRAM grows between phases and spills into shared memory, and
7B on an 8 GB GPU takes ~13 minutes per image. Without this block users with 8–12 GB GPUs either cannot use 7B or end up
with a starved system after the upscale.

## What we do (steps)
1. **Pipeline import first.** Move the `seedvr2_vendor` imports to the top of `run()` (before the base image and the SD
   unload); an import error returns at once, nothing generated or unloaded.
2. **Forge device.** Use Forge's device (`modules.devices.device`) instead of hard-coded `cuda:0`.
3. **Pure helpers in `seedvr2_neo/`** (no Forge imports, testable): building the pipeline arguments from UI values
   (offload devices, BlockSwap config, cache flags), the low-VRAM warning text (model size × GPU total memory), the cache
   key (absolute path, not file name only).
4. **"After run" choice in the UI:** unload SeedVR2 completely (VRAM and RAM freed) / keep in RAM for a faster next run.
   Unload = `GlobalModelCache` entries removed, runner and context dropped, `gc` + empty cache.
5. **Real Force Reload:** clears the context **and** the `GlobalModelCache` DiT/VAE/runner entries.
6. **Context reset on error / interrupt:** on any exception the cached context is dropped (no partial state reused);
   the `finally` block deletes the keys the phases actually create.
7. **Release reserved VRAM between phases:** empty the allocator cache after encode and after the DiT step (via the
   phase boundaries in `run()`; a vendored patch only if the boundary is not reachable from `scripts/`).
8. **BlockSwap / offload in the UI:** "Blocks to swap" slider (0 = off; max 36, the pipeline clamps to the model's
   block count), "I/O layers" checkbox; when either is on, DiT and VAE offload device = CPU (DiT: required by upstream;
   VAE: idle during the DiT step).
9. **7B warning on small GPUs:** a visible note under the model dropdown when a 7B DiT is selected and the GPU has less
   than 11 GiB total memory, recommending BlockSwap. The recommended block count is added after the live check.
10. **"Unload SD Checkpoint" hint:** the label/info says the checkpoint is reloaded on the next generation.
11. Tests for the helpers in `tests/seedvr2_neo/`; update `05-DECISIONS`, `07-RISKS`, `09-PROCESSES`, `02-ARCHITECTURE`,
    `external-apis/` (Forge memory facts), local patches list if `seedvr2_vendor/` is touched.
12. **Live check by the owner** on an 8 GB GPU without bf16: see acceptance criteria.

## Acceptance criteria
- [ ] A broken pipeline import (simulated) returns an error **before** the base image is generated and before the SD unload — shown by a test or a console log.
- [ ] No `cuda:0` / `"0"` device literal left in `scripts/seedvr2_script.py` (`grep`).
- [ ] "After run: unload" — after the run, Task Manager shows the Forge process RAM back near its pre-run level (± 1 GB) and GPU memory near the pre-run level; console logs the unload.
- [ ] "After run: keep in RAM" — a second run with the same models skips model loading (console: cache hit).
- [ ] Force Reload — the console shows the DiT and VAE loaded from disk again on a run with Force Reload on.
- [ ] Replacing the selected file with another of the same name in the other model folder is picked up (cache key = absolute path) — test.
- [ ] After a simulated exception inside a phase, the next run starts with a fresh context — test; live: an interrupt mid-run, then a normal run gives a clean image.
- [ ] 7B fp8 on the 8 GB GPU with BlockSwap on: the DiT step at 1080px is at least 3× faster than the current 780 s, with a clean image (exact block count recorded in `07-RISKS.md`).
- [ ] With a 7B DiT selected on a GPU below the threshold, the UI shows the warning; with 3B it does not.
- [ ] 3B fp8 and 3B GGUF still give clean images on the default settings (regression).
- [ ] All tests pass (`uv run … python -m pytest tests -q`), new helpers covered in `tests/seedvr2_neo/`.

## Which checks are needed on `/block-done`
Always: `code-auditor`, `fresh-eyes-reviewer`, `plan-drift-checker`. Additionally:
- [x] `integration-checker` — the block must run in Forge; new UI arguments change `run()`'s signature and saved UI values
- [ ] `security-auditor` — no user data, no external requests, no new file writes
- [x] `stress-tester` — UI input (slider bounds, block count above the model's, BlockSwap with GGUF), OOM / interrupt mid-phase

## Dependencies
B05 (model discovery) — done. The owner's live runs in Forge Neo on an 8 GB GPU (steps 12, criteria).

## Out of scope for this block
- Every image of a batch, infotext, save settings, RGB conversion — B07.
- Interrupt inside the DiT step, Forge progress bar, color-correction / attention choice, title — B08.
- Forge stubs for testing `run()` itself, CI — B09.
- fp16 compute instead of emulated bf16 on GPUs without bf16 (`07-RISKS.md` §3) — needs measuring; moved to `06-IDEAS.md`.
- Partial SD offload (`free_memory` for just the needed amount instead of a full checkpoint unload) — `06-IDEAS.md` unless the fact-check shows it is trivial.
- Bounded GGUF header parser (`07-RISKS.md` §1).

## Hidden assumptions
- Upstream: caching needs an offload device; BlockSwap and `swap_io_components` need a DiT offload device
  (`external-apis/seedvr2-upstream.md`). With caching on, `cleanup_dit` / `cleanup_vae` move the models to the offload
  device or to CPU (`seedvr2_vendor/optimization/memory_manager.py`).
- BlockSwap works with GGUF-quantized DiT — not verified; checked by the live run (`07-RISKS.md` §3).
- Forge Neo (`external-apis/forge-neo.md` → "Memory"): the launch GPU flag sets `CUDA_VISIBLE_DEVICES`, so
  `modules.devices.device` is the right device; `devices.torch_gc()` = synchronize + `empty_cache` + `ipc_collect`;
  Forge measures free VRAM from the driver and has **no hook** to ask an extension to release memory — SeedVR2 must
  free its own memory before SD loads again. Default allocator is PyTorch's native one (`--cuda-malloc` is opt-in).
- `run()` is still not covered by tests (Forge stubs are B09): the import-before-work order and the phase-boundary
  cache release are checked by review and the live run.

## Live check for the owner (step 12)
Run Forge with the extension on this branch, an 8 GB GPU, 1080px output. Watch the console and Task Manager
(Performance → GPU "Dedicated GPU memory", Memory "In use").
1. 3B fp8, defaults (After run = Unload): clean image; console ends with "Models unloaded, memory released."; RAM and
   dedicated GPU memory back near the levels before the run; during the run RAM does not jump by the model's size at
   the end of Phase 2 (the DiT is freed on the GPU, not copied to RAM first).
2. Same, After run = Keep in RAM, two runs: the second has no model loading in the console ("Reusing cached runner" /
   cache hit); console says "Models kept in RAM for the next run."
3. Keep in RAM + Force Reload on: the models are loaded from disk again.
4. Interrupt during Phase 2, then a normal run: clean image.
5. Select a 7B DiT: the warning note appears; select 3B: it disappears.
6. 7B fp8 with BlockSwap 16, then 24, then 32 blocks (I/O layers off, then on for the best one): DiT step time from the
   console (Show Debug Logs on for phase timings), clean image. Goal: at least 3× faster than 780 s.
7. 3B GGUF with BlockSwap 16: clean image (or the error, if it does not work).
8. Watch for "VRAM overflow … paged to system RAM" lines in the debug log (Show Debug Logs on) in run 1.

## Live check results (8 GB GPU without bf16, 1080px, tile 512 unless noted)
Runs 1–2 by the owner through the UI, the rest through Forge's API (img2img with the script) with the owner's flags.
- 1 ✅ Unload: "DiT not cached - skipping CPU movement"; RAM did not grow at the end of Phase 2; the DiT's 3.2 GB VRAM
  is returned by the between-phase `gc.collect()` + empty-cache (still allocated right after the phase); GPU back to
  ~0.6 GB used after the run.
- 2 ✅ Keep in RAM: second run "Reusing cached runner / DiT / VAE", no disk reads; DiT CPU→GPU 11.8 s vs 28 s from disk.
- 3 ✅ Force Reload: "Cached models dropped…", both models read from disk, the old copies freed (process RAM 1.2 GB).
- 4 ✅ Interrupt in Phase 2 (Keep in RAM): "Interrupted", then "Models unloaded"; next run clean.
- 5 ⬜ 7B warning in the UI — owner to check.
- 6 ✅ 7B fp8, DiT step (was 780 s): BlockSwap 16 → 34.5 s, 8 → 21 s (0 GB free), 4 → 297 s (spill), 8 + I/O → 22.8 s;
  image clean. Without `--cuda-malloc`: 16 → 42 s, 8 → 122 s. Recommended: 16 (in the warning).
- 7 ✅ 3B GGUF Q6_K + BlockSwap 16: works, image clean (DiT 17 s; without BlockSwap 7 s — 3B needs none).
- 8 VAE decode spill: with `--cuda-malloc` (cudaMallocAsync) the decode reserves 8.7 GB at a 5.2 GB peak, pages
  0.7–2.9 GB and takes 25–137 s; without it 6.7 GB reserved, no overflow, 12.4 s stable. Tile 512 vs 1024 (with
  `--cuda-malloc`): encode 5.7 s vs 16 s and no spill; decode spill 0.9 vs 2.9 GB.

## Open questions
None.
