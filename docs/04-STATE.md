# PROJECT STATE — seedvr2-webui-neo-extension

## Where we stopped

**B09 work done, `/block-done` next** — branch `feature/b09-forge-stubs-ci` (pushed with the owner's consent; CI green:
run 44610246289). B07 is merged into `main` and pushed.

What B09 did: Forge/Gradio/pipeline stand-ins in `tests/scripts/forge_stubs.py` and tests that drive `Script.run()`
(txt2img batches, img2img, image modes, saving, errors, interrupts, SD unload, model lifecycle); the script split
(530 → 317 lines) into `seedvr2_neo/upscale_run.py` (pipeline run, the only torch user in `seedvr2_neo/`) and
`run_results.py` (result images); `.github/workflows/tests.yml` — pytest on every push (owner chose minimal CI, no ruff).
No behavior change intended.

Live run (Forge API, 8 GB GPU without bf16, 3B GGUF): txt2img batch of 2 fine; img2img stalled for many minutes in the
VAE phase — the same on `main`, so recorded as 🟠 R9 (default tile 1024 with `--cuda-malloc`), not a B09 regression.
The owner's local Forge is back on `main`, stopped. For the API checks a self-contained SDXL checkpoint was set per request
(`override_settings`): the owner's selected one needs a separate VAE module.

`seedvr2_vendor/` and the `.pt` embeddings match upstream numz/ComfyUI-SeedVR2_VideoUpscaler `4490bd1`, plus the local
patches in `external-apis/seedvr2-upstream.md`. Known problems: `07-RISKS.md` (top: 🟠 R9 VAE tile 1024 stall).

Tests: 1658 passed, 1 skipped (POSIX-only), 3 xfailed. Run: `uv run --no-project --python 3.10 --with-requirements
requirements-dev.txt python -m pytest tests -q` (pulls CPU torch on Windows). CI: GitHub Actions on every push.

## Completion percentage

**78%** (18 of 23) — counted by `plan-drift-checker` on closing B07. 2 blocks open.

## Next steps

1. `/block-done` for B09: reviewers (code-auditor, fresh-eyes on the split, integration-checker, plan-drift), then the
   owner's consent to merge into `main`.
2. Propose via `/idea`: VAE tile 512 as default (🟠 R9).
3. B08 — settings and feedback (progress bar, interrupt inside the DiT step, SeedVR2 keys read back, title).

Agreed with the owner: REQ-1…REQ-6, simplified Git Flow, ComfyUI files kept as in upstream (Forge Neo only),
hardware target — 8 GB GPU without bf16, the whole repository under Apache-2.0, no grid of upscaled images. The fork is
public: `main` is what users get, pushes only with consent.

## Open questions for the owner

None.
