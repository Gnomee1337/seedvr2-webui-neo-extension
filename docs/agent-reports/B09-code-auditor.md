# B09 — code-auditor report

Scope: `git diff main...HEAD` on `feature/b09-forge-stubs-ci` (5 commits, 40f7216..62d2102): `scripts/seedvr2_script.py`,
`seedvr2_neo/upscale_run.py`, `seedvr2_neo/run_results.py`, `tests/scripts/*`, `tests/seedvr2_neo/test_upscale_run.py`,
`tests/seedvr2_neo/test_run_results.py`, `.github/workflows/tests.yml`, `requirements-dev.txt`.

Test run: `uv run --no-project --python 3.10 --with-requirements requirements-dev.txt python -m pytest tests -q -p no:cacheprovider`
→ **1658 passed, 1 skipped, 3 xfailed**. The safety-net tests (`tests/scripts/`) are byte-identical between the
pre-split commit 40f7216 and HEAD (`git diff 40f7216 HEAD -- tests/scripts` is empty), so they passed unchanged on both sides.

Mutation checks were run in a throw-away copy (`git archive HEAD`) in the scratchpad, removed afterwards.

**Totals: BLOCKER 0, IMPORTANT 0, MINOR 9.**

## Behavior equivalence of the split (main → HEAD) — no change found

Compared line by line:
- `_upscale_images` / `_upscale_pass` / `_upscale_tensor` → `UpscaleRun.upscale_images` / `_upscale_pass` / `_upscale_tensor`:
  same order (`get_global_cache` and `prepare_run` outside `try`; `Debug`, passes, per-pass interrupt check, `Done.` inside;
  `finish_run(..., failure is None)` in `finally`), identical kwargs to all six pipeline calls (`"lab"`, `"sdpa"` now
  constants with the same values), identical console texts including the `(Tiled)` notes, same exception classes caught
  (`UpscaleInterrupted` replaces `SeedVR2Interrupted`; the vendored phases only log and re-raise, they never match by name).
- `check_interruption` → bound method reading `hooks.is_interrupted()`; `forge_hooks()` passes
  `lambda: shared.state.interrupted`, evaluated lazily (the mid-phase interrupt test would fail on an eager capture).
- `devices.device` is now read once per run in `forge_hooks()` instead of in each pass; it is a module variable fixed
  at Forge start (`forge-neo.md`), so the value is the same.
- `build_result`: embedding now happens for all images before any save (was interleaved per image). The image that is
  saved is always embedded before its own save in both versions, and `save_image` only marks the image it saves, so
  files, infotexts, comment text and `Processed` are identical. `shared.opts.enable_pnginfo` read once instead of per image — same value.
- `describe_txt2img_inputs` / `describe_img2img_inputs` → `describe_generated_inputs` / `describe_uploaded_inputs`: same fields.
- Removed imports (`traceback`, `SimpleNamespace`, `Any`, `Tuple`, `FormRow`, `transforms`, `numpy`) are all unused in the
  new script; no unused imports remain in the changed files.

## Findings

### MINOR-1 🟡 R4 (S2×L2) — `forge_hooks` wiring (device, memory release) is not checked by any test
`scripts/seedvr2_script.py:72-76`. The script tests use a CPU stub device and never look at what reaches
`setup_generation_context` or at `forge.torch_gc_calls` (the counter at `tests/scripts/forge_stubs.py:101,121` is
never asserted). Proven by mutation: changing line 75 to
`HostHooks(device=torch.device("cpu"), release_memory=lambda: None, ...)` → all 49 script tests pass.
Scenario: a later edit passes the wrong device or a no-op release → CI green, users with a GPU get SeedVR2 on CPU (or on
the wrong GPU) and no VRAM release between phases. Same for removing `devices.torch_gc()` after the SD unload
(`scripts/seedvr2_script.py:255`, mutant also survives).
Fix: give the stub `devices.device` a distinct sentinel (e.g. `torch.device("meta")` or an object) and assert it in the
`setup` kwargs the fake pipeline records; assert `torch_gc_calls` after a run with `unload_sd=True`.

### MINOR-2 🟢 R2 (S1×L2) — "Show Debug Logs" path has no test
`seedvr2_neo/upscale_run.py:124` and every `if not debug_mode` / `if debug_mode` print. All tests run with
`debug_mode=False`; the stub `Debug` records `enabled` but nobody reads it. Mutation `Debug(enabled=not debug_mode)` →
all script and `test_upscale_run` tests pass. Scenario: the flag gets inverted → vendored debug logging floods the console
on every normal run and disappears when the user asks for it.
Fix: one script test with `debug_mode=True` asserting the fake `Debug` got `enabled=True` (and `False` by default).

### MINOR-3 🟢 R2 (S1×L2) — tautological assertion in the default-unload test
`tests/scripts/test_seedvr2_script.py:283-286`. With the default "Unload SeedVR2" plan the fake `prepare_runner`
(`forge_stubs.py:313-315`) never stores anything, so `pipeline.cache.cached == set()` holds whether or not the run
removes models at the end; only the `dit_cache is False` line checks something. The test name promises "models are
unloaded after a run" but would pass if `finish_run` were skipped for "Unload".
Fix: assert the end-of-run removal itself (`pipeline.cache.removed == CACHE_IDS` or equivalent), or rename the test to
what it checks.

### MINOR-4 🟢 R2 (S1×L2) — interrupt in a later pass is not covered
`seedvr2_neo/upscale_run.py:127-134`, `UpscaleOutcome` docstring (`:72-73`) promises images of finished passes are kept
"even then". Only the error variant is tested (`test_error_in_a_later_pass_keeps_the_images_already_upscaled`); the
interrupt variant is not, though `FakePipeline` supports it (`interrupt_in` + `only_in_pass`). Scenario: someone resets
`upscaled` in the `UpscaleInterrupted` branch → CI green, an interrupted 2-pass batch loses the first pass's results.
Fix: copy the later-pass error test with `interrupt_in="upscale", only_in_pass=1`.

### MINOR-5 🟢 R1 (S1×L1) — stub `p` claims Forge defaults it does not have; dead `FormRow` stub
`tests/scripts/forge_stubs.py:152-163`: docstring "with Forge's defaults", but `outpath_samples="outputs"` (Forge and
`forge-neo.md`: `None`) and `width/height=8` (Forge: 512). No current test depends on it, but a test of
`save_result_image` with a default `p` would pass against a path Forge never provides.
`tests/scripts/forge_stubs.py:243-247`: `FormRow` stub, while the script no longer imports `FormRow` (removed in this
block); `docs/external-apis/forge-neo.md:294` still lists it under "Names `scripts/seedvr2_script.py` uses" and the card
(`docs/roadmap-blocks/B09-forge-stubs-ci.md:59`) says "imported but unused".
Fix: make the docstring say which defaults are deliberately different (or use `None`/512 and set them in fixtures);
drop the `FormRow` stub and its line in the "Names the script uses" section.

### MINOR-6 🟢 R1 (S1×L1) — function and module size over the guideline
- `seedvr2_neo/upscale_run.py:157-228` `_upscale_tensor` — ~70 lines (guideline 30–40); rewritten in this block, so it
  falls under "new and rewritten code". Natural split: `_prepare(ctx, runner)` and `_run_phases(...)`.
- `scripts/seedvr2_script.py:218-264` `run()` — ~47 lines.
- `tests/scripts/forge_stubs.py` — 408 lines (guideline ~300) mixing Gradio, Forge `modules.*` and the pipeline fakes;
  the pipeline part (`:251-362`) is a separate module by meaning (`pipeline_stubs.py`).

### MINOR-7 🟢 R1 (S1×L1) — type hints lost or loose on public code
`scripts/seedvr2_script.py:128` `build_result(..., params, ...)` lost its `Dict[str, Any]` annotation in the split;
`seedvr2_neo/upscale_run.py:65` `HostHooks.device: Any` although the module imports torch (`torch.device` is the type);
`_upscale_tensor` has no return type. code-style: "Type hints on all public functions; `Any` only at the boundary with foreign JSON".
Fix: `params: Mapping[str, Any]`, `device: torch.device`, `-> torch.Tensor`.

### MINOR-8 🟢 R1 (S1×L1) — PEP 8 spacing and a history-style docstring in `forge_hooks`
`scripts/seedvr2_script.py:70-72`: one blank line between `MODEL_LIFECYCLE = ...` and `def forge_hooks` (two elsewhere).
`:73-74`: "the `.stopping` check was removed on purpose" describes history (protocol §11), and repeats the knowledge
already stated in `UpscaleRun.check_interruption` (`seedvr2_neo/upscale_run.py:107-108`, which words it as current state).
Fix: add the blank line; reduce the `forge_hooks` docstring to what it returns.

### MINOR-9 🟢 R1 (S1×L1) — docs/comments that no longer match the code
- `CLAUDE.md:43`: "the script itself is not covered yet (Forge stubs planned)" — B09 covers it.
- `tests/scripts/test_seedvr2_script.py:29`: "the script truncates to 8 bits" — truncation is now in
  `seedvr2_neo/upscale_run.py:94` (`tensor_to_images`).
- `requirements-dev.txt:2`: "The tests of the Forge script need torch and torchvision" — `tests/seedvr2_neo/test_upscale_run.py`
  needs them too.

## Checked and fine
- `seedvr2_neo/` imports no Forge or Gradio; torch/torchvision only in `upscale_run.py` (as `02-ARCHITECTURE.md` states).
  `import_pipeline` stays lazy, so a broken vendored install is still reported in the UI.
- Stubs vs `forge-neo.md` "Names the script uses": `shared` (models_path, opts defaults, state fields), `devices.device` /
  `torch_gc()`, `sd_models.unload_model_weights` reachable via `sys.modules` and the package attribute, `images.save_image`
  full signature + `(fullfn, txt_fullfn)` + `already_saved_as`, `Processed.__init__` signature and `comments` join,
  `p.comment` dict, `ScriptRunner.scripts` list, `Script` class attributes and `elem_id`, `ToolButton` signature,
  `tooltip=` accepted, `gr.update` / `.click` / `.change`: all match (except MINOR-5).
- Each scenario of card step 2 has a script test; error infotext carries only the exception type (path in the message
  is checked not to leak).
- `test_upscale_run.py` and the new `test_run_results.py` tests check real behaviour (wrong output count → failure,
  release after 3 phases, copies keep `already_saved_as`, embed flag off leaves `info` alone).
- CI: CPU torch installed before `requirements-dev.txt`, so the unpinned `torch` line is already satisfied and the CUDA
  build is not pulled; `permissions: contents: read`; timeout set.
