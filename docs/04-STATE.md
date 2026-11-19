# PROJECT STATE — seedvr2-webui-neo-extension

## Where we stopped

**B10 closed** on `feature/b10-vae-tile-512` (CI green); **not merged into `main` yet — waiting for the owner's consent.**

What B10 changes for users: the VAE tile size defaults to 512 (was 1024) on a slider renamed "VAE Tile Size", so
Forge's saved `ui-config.json` value no longer overrides it — existing users get 512 once, a value set on purpose
included (README says so). The tile overlap is capped at half the tile (a saved overlap of 512 next to the new 512
tile would have meant thousands of tiles). Live, 8 GB GPU without bf16, 3B GGUF, img2img 512×384 → 1364×1024: 128 s
with `--cuda-malloc` (the decode still spills, 🟡 R6), 68 s without; both images clean; the old 6–22+ minute stall is gone.

Found on the way (recorded in `external-apis/forge-neo.md`, 🟢 in `07-RISKS.md`): Forge does not pad a short API
`script_args` with defaults — the request fails with HTTP 500, so API callers send all arguments.

The owner's local Forge is on `main`, stopped. For API checks a self-contained SDXL checkpoint is set per request
(`override_settings`): the owner's selected one needs a separate VAE module.

`seedvr2_vendor/` and the `.pt` embeddings match upstream numz/ComfyUI-SeedVR2_VideoUpscaler `4490bd1`, plus the local
patches in `external-apis/seedvr2-upstream.md`. Known problems: `07-RISKS.md` (top: 🟡 R6 VAE decode spill with
`--cuda-malloc`).

Tests: 1674 passed, 1 skipped (POSIX-only), 3 xfailed. Run: `uv run --no-project --python 3.10 --with-requirements
requirements-dev.txt python -m pytest tests -q` (pulls CPU torch on Windows). CI: GitHub Actions on every push.

## Completion percentage

**68%** (22 of 32). 3 blocks open.

## Next steps

1. Merge `feature/b10-vae-tile-512` into `main` — only with the owner's consent.
2. B11 — output size: scale factor ×1.0–×4.0 and long edge in px; default stays 1080 px shortest edge; no 6K/8K
   until B12. Assessment: `agent-reports/idea-integrator-scale-presets.md` (defaults: the factor applies to the final
   txt2img image, after Hires. fix; inputs under 512 px are raised to 512 with a note). A changed slider default
   reaches existing users only under a new label (`05-DECISIONS.md`, VAE tile decision).
3. B08 — settings and feedback, plus a simple main UI and GPU profiles (8 / 12 / 24 GB).
4. B12 — DiT tiling with feather blending (4K+ on 8 GB), then 6K/8K options.

Agreed with the owner: REQ-1…REQ-6, simplified Git Flow, ComfyUI files kept as in upstream (Forge Neo only),
hardware target — 8 GB GPU without bf16, the whole repository under Apache-2.0, no grid of upscaled images. The fork is
public: `main` is what users get, pushes only with consent.

## Open questions for the owner

1. Merge B10 into `main` (users get it on their next "Check for updates")?
