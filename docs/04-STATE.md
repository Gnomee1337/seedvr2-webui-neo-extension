# PROJECT STATE — seedvr2-webui-neo-extension

## Where we stopped

**B09 closed** and merged into `main` with the owner's consent (CI green on Python 3.10 and 3.13).

What B09 changes: no user-visible behavior change. `Script.run()` is tested against stand-ins of Forge, Gradio and the
pipeline (`tests/scripts/`), including every keyword sent to the vendored pipeline checked against its source; the script
is split (530 → ~320 lines) into `seedvr2_neo/upscale_run.py` (pipeline run, the only torch user in `seedvr2_neo/`) and
`run_results.py` (result images); `.github/workflows/tests.yml` runs pytest on every push, Python 3.10 and 3.13, CPU torch
(owner chose minimal CI: no linter).

Live run (Forge API, 8 GB GPU without bf16, 3B GGUF): txt2img batch of 2 fine; img2img stalled for many minutes in the
VAE phase — the same on `main`, so recorded as 🟠 R9 (default tile 1024 with `--cuda-malloc`), not a B09 regression.
The owner's local Forge is on `main`, stopped. For API checks a self-contained SDXL checkpoint is set per request
(`override_settings`): the owner's selected one needs a separate VAE module.

`seedvr2_vendor/` and the `.pt` embeddings match upstream numz/ComfyUI-SeedVR2_VideoUpscaler `4490bd1`, plus the local
patches in `external-apis/seedvr2-upstream.md`. Known problems: `07-RISKS.md` (top: 🟠 R9 VAE tile 1024 stall).

Tests: 1664 passed, 1 skipped (POSIX-only), 3 xfailed. Run: `uv run --no-project --python 3.10 --with-requirements
requirements-dev.txt python -m pytest tests -q` (pulls CPU torch on Windows). CI: GitHub Actions on every push.

## Completion percentage

**65%** (21 of 32) — B09 closed at 91% (21 of 23), then the owner added B10 (1), B11 (2), B12 (5) and grew B08 (2 → 3).
4 blocks open.

## Next steps

1. B10 — VAE tile 512 as the default (🟠 R9). Agreed with the owner: own block before B08; rename the slider so
   existing users get 512 too (one reset of a hand-set value); Claude does the live re-measure. Assessment:
   `agent-reports/idea-integrator-tile-512-default.md`; fact-check Forge Neo's saved UI values (`ui-config.json`) first.
2. B11 — output size: scale factor ×1.0–×4.0 and long edge in px; default stays 1080 px shortest edge; no 6K/8K
   until B12. Assessment: `agent-reports/idea-integrator-scale-presets.md` (defaults: the factor applies to the final
   txt2img image, after Hires. fix; inputs under 512 px are raised to 512 with a note).
3. B08 — settings and feedback, plus a simple main UI and GPU profiles (8 / 12 / 24 GB).
4. B12 — DiT tiling with feather blending (4K+ on 8 GB), then 6K/8K options.

Agreed with the owner: REQ-1…REQ-6, simplified Git Flow, ComfyUI files kept as in upstream (Forge Neo only),
hardware target — 8 GB GPU without bf16, the whole repository under Apache-2.0, no grid of upscaled images. The fork is
public: `main` is what users get, pushes only with consent.

## Open questions for the owner

None.
