# PROJECT STATE — seedvr2-webui-neo-extension

## Where we stopped

**B07 closed** (correct results) on branch `feature/b07-correct-results`; merge into `main` and push wait for the
owner's consent (`main` reaches users on their next "Check for updates").

What B07 changes for users: txt2img upscales every image of a batch (not the grid; no grid of the results), img2img
every input (Batch tab too); grayscale / palette / 16-bit / CMYK inputs become RGB, transparency kept; each result keeps
its generation parameters plus `SeedVR2 …` keys (also inside unsaved images); files only with "Always save" on (or the
API's `save_images`), nothing otherwise; "Save Extra Copy of Original" now saves only img2img inputs; a failed or
interrupted upscale returns the images as they were with `SeedVR2 Error` and a note; API values clamped to the slider
ranges; batches run in passes of up to 16 MP of output. New helpers: `seedvr2_neo/run_inputs.py`, `run_results.py`,
`upscale_settings.py`.

Live-checked through Forge's API on an 8 GB GPU without bf16 (3B GGUF): batches with grid, odd image modes, saving
on/off, interrupt, PNG Info read-back; img2img Batch tab by the owner. Claude runs such checks itself (Forge launched
with `--api`; private setup in Claude's memory). The owner's local Forge has the extension on test branch `b07-test`.

`seedvr2_vendor/` and the `.pt` embeddings match upstream numz/ComfyUI-SeedVR2_VideoUpscaler `4490bd1`, plus the local
patches in `external-apis/seedvr2-upstream.md`. Known problems: `07-RISKS.md` (top: VAE decode spills with
`--cuda-malloc`, multiplied by batches).

Tests: 1594 passed, 1 skipped (POSIX-only), 3 xfailed (known bugs / limitations, `07-RISKS.md`). pytest is not
installed in the system Python; run `uv run --no-project --python 3.10 --with-requirements requirements-dev.txt python -m pytest tests -q`.
CI: none.

## Completion percentage

**78%** (18 of 23) — counted by `plan-drift-checker` on closing B07. 2 blocks open.

## Next steps

1. B09 — Forge stubs for testing `run()` + CI (also split `scripts/seedvr2_script.py`, ~520 lines).
2. B08 — settings and feedback (progress bar, interrupt inside the DiT step, SeedVR2 keys read back, title).

Agreed with the owner: REQ-1…REQ-6, simplified Git Flow, ComfyUI files kept as in upstream (Forge Neo only),
hardware target — 8 GB GPU without bf16, the whole repository under Apache-2.0, no grid of upscaled images. The fork is
public: `main` is what users get, pushes only with consent.

## Open questions for the owner

None.
