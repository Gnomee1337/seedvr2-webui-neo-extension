# B06 integration check

Scope: `git diff main...HEAD`, branch `feature/b06-vram-lifecycle`. BLOCKER: none. IMPORTANT: none.

## Checked and fitting
- `ui()` returns 16 components, `run()` takes 16 arguments in the same order; the first 13 match `main`, the 3 new ones
  default to Unload / 0 / False, so old API payloads keep working.
- Without `seedvr2_vendor` (meta-path blocker + stubs for Forge, torch, gradio): the script imports, `ui()` builds,
  `describe_vram_warning` returns "", `run()` returns `Error: …` before `process_images` and the SD unload.
- Keyword arguments of `setup_generation_context`, `prepare_runner`, `compute_generation_info` and the four phases
  match the vendored signatures; text embeddings load lazily in Phase 2.
- `remove_dit` / `remove_vae` take `{"node_id": …}`, return bool, and drop the `forge_dit+forge_vae` runner template.
- BlockSwap config keys match `blockswap.py`; the `"cpu"` string works where upstream passes `torch.device`
  (compared by `str()` or passed to `.to()`).
- The tracked path `os.path.join(scan.folder, name)` is the path `find_model_file` returns (caller's folder first).
- Local patch in `memory_manager.py`: cached path and MPS unchanged; Unload skips the CPU copy; listed in
  `seedvr2-upstream.md`.
- `MODEL_LIFECYCLE` shared by both tabs matches the process-wide vendored cache.
- Gradio 4.40: `Radio`/`Checkbox` with `info`, `Markdown(visible=…)` valid; refresh fires `dit_model.change`.
- Tests: 329 passed, 1 skipped, 2 xfailed.

## MINOR findings
1. 🟢 R2 `seedvr2-upstream.md` said the script "caches with no offload — models stay in VRAM"; in fact `cleanup_dit` /
   `cleanup_vae` fall back to the CPU (live run 2: DiT CPU→GPU 11.8 s); the patch line still said "to be checked".
2. 🟢 R2 "Keep in RAM" + error/interrupt in Phase 2: the phase's `finally` copies the DiT to RAM (`cache_model=True`),
   then `finish_run` drops it — a RAM spike and a few seconds after Interrupt; result correct.
3. 🟢 R2 No test runs our code against the real `GlobalModelCache` / `validate_blockswap_config`; an upstream sync could
   change them with all tests green. Fix in B09.
4. 🟢 R2 Not run live: Keep in RAM + BlockSwap; Keep in RAM with the block count changed between runs
   (`_handle_blockswap_change`); interrupt in Unload mode.
5. 🟢 R1 Keep in RAM → Unload with the same files reloads from disk (the vendored cache removes entries when caching is off).

Already known: `import_pipeline` catches only `ImportError`; BlockSwap on CPU-only Forge fails in `validate_blockswap_config`.
