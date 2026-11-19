# B09 — integration check (script split, Forge stubs, CI)

Scope: `git diff main...HEAD` on `feature/b09-forge-stubs-ci`. No BLOCKER and no IMPORTANT findings; 3 MINOR.

## What was verified

| Seam | Result | How |
|---|---|---|
| Names the script imports from `seedvr2_neo.*` exist | OK | AST walk of `scripts/seedvr2_script.py` imports + `hasattr` on every name |
| Signatures of the moved calls | OK | `UpscaleRun(pipeline, MODEL_LIFECYCLE, forge_hooks()).upscale_images(images, request, force_reload, debug_mode)` matches `seedvr2_neo/upscale_run.py:101,112`; `finish_images(..., shared.opts.enable_pnginfo)` matches `run_results.py:162`; `not_upscaled_comment`, `describe_generated_inputs`, `describe_uploaded_inputs` match their call sites (`scripts/seedvr2_script.py:146-147,301,304`) |
| Removed names (`check_interruption`, `SeedVR2Interrupted`, `describe_txt2img_inputs`, `describe_img2img_inputs`, `embed_infotext`, module-level `import_pipeline` in the script) | no callers left | repo-wide grep outside tests |
| `scripts/` holds only `seedvr2_script.py` (tracked); `__pycache__` ignored | OK | `git ls-files scripts`, `.gitignore:16` |
| `seedvr2_neo/` imports neither `gradio` nor `modules`; torch/torchvision only in `upscale_run.py` | OK | grep + importing every other `seedvr2_neo` module and checking `sys.modules` (`torch`, `gradio`, `modules` absent) |
| `import_pipeline` imports the same vendored names as `main` | identical | `seedvr2_neo/upscale_run.py:37-46` vs `git show main:scripts/seedvr2_script.py` lines 84-93 |
| Keyword arguments vs the real vendored signatures | all bind | real `import_pipeline()` (with psutil/omegaconf/einops/diffusers/rotary-embedding-torch added ad hoc), then `inspect.signature(fn).bind(...)` for every `pipeline.*` call in `upscale_run.py:167-227`: `setup_generation_context`, `prepare_runner`, `compute_generation_info`, `encode_all_batches`, `upscale_all_batches`, `decode_all_batches`, `postprocess_all_batches`; `Debug(enabled=)` too. Values passed are the same as on `main` (`"lab"`, `"sdpa"` became constants) |
| Progress callback | OK | bound method `check_interruption(*args, **kwargs)` accepts the vendor's `(int, int, int, str)` |
| `Script.run` order vs `ui()` return list | unchanged from `main`, 16 = 16, same order | diff + `tests/scripts/test_seedvr2_script.py:37` asserts it |
| Workflow deps / directory | OK | `tests.yml` runs `python -m pytest tests` from the checkout root; pytest picks `tests/pytest.ini` (`pythonpath = ..`); every third-party import in `tests/` (`pytest`, `PIL`, `numpy`, `gguf`, `torch`) is in `requirements-dev.txt` (+ CPU torch/torchvision installed first); `_winapi` is Windows-only and guarded |
| Python 3.10 syntax of the changed files | OK | `ast.parse(..., feature_version=(3, 10))` on the script, `upscale_run.py`, `run_results.py` and the new tests; nothing 3.11+-only (no `Self`, `match`, `ExceptionGroup`, PEP 695) |
| Full test run (Python 3.10) | **1658 passed, 1 skipped, 3 xfailed** | `uv run ... python -m pytest tests -q -p no:cacheprovider` |

## Findings

### MINOR 1 — 🟡 R6 (S3×L2): the fake pipeline accepts any keyword, so a signature drift passes every test
`tests/scripts/forge_stubs.py:308-335` — `FakePipeline.setup_generation_context(**kwargs)`, `prepare_runner(**kwargs)`,
`compute_generation_info(ctx, images, **kwargs)`, `*_all_batches(..., **kwargs)`.
Scenario: an upstream sync renames a parameter (e.g. `input_noise_scale`) or someone mistypes a keyword in
`seedvr2_neo/upscale_run.py:176-227`; all 1658 tests and CI stay green; in Forge every run ends in
`Runtime Error: TypeError`, images come back not upscaled (S3 — no upscale for anyone). Today all calls bind (checked by hand).
Fix: a test that reads the parameter names of the seven vendored functions from
`seedvr2_vendor/core/generation_utils.py` / `generation_phases.py` with `ast` (no import, so no psutil/diffusers needed in
CI) and checks that the keywords the fake recorded during a run are a subset of them; or give the fake the exact signatures.

### MINOR 2 — 🟢 R3 (S3×L1): CI never imports the real vendored pipeline
`.github/workflows/tests.yml:1-2,25-30`, `requirements-dev.txt` — the pipeline is always stubbed via `sys.modules`
(`tests/scripts/forge_stubs.py:347-362`), and `psutil`, `omegaconf`, `einops`, `diffusers`, `rotary-embedding-torch`
are not installed, so `import_pipeline()` (`seedvr2_neo/upscale_run.py:33`) is never executed for real.
Scenario: a vendored file is left out of a commit or a vendored import breaks; CI is green; users get
"Error: cannot import the SeedVR2 pipeline" on every run. The workflow header claims it catches "a file left out of a commit" — true only for files the tests import.
Fix: either an import-only test of `seedvr2_vendor.core.generation_utils` / `generation_phases` with those packages in a
separate CI step (CPU, no models), or at least an `ast`-level test that every module `import_pipeline` names exists in the repo; or soften the workflow comment.

### MINOR 3 — 🟢 R2 (S2×L1): CI runs only Python 3.10
`.github/workflows/tests.yml:24`. Forge Neo installs may run 3.11–3.13; nothing in the new code is version-sensitive today,
but a 3.12+ deprecation (e.g. in a dependency) would not show. Fix: a `python-version` matrix `["3.10", "3.13"]` (doubles ~1 min of CI).

## Not verified
- That `actions/checkout@v7` / `actions/setup-python@v7` exist — taken from `docs/external-apis/github-actions.md` (sources
  listed there; it also cites a green first CI run). No network check from here.
- Real Forge Neo load (no Forge in this check); module-level imports of the script are the same as on `main` minus
  `FormRow`, `traceback`, `numpy`, `torchvision` (moved into `upscale_run.py`, still loaded at script import).
