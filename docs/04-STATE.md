# PROJECT STATE — seedvr2-webui-neo-extension

## Where we stopped

**B08 closed** and merged into `main` with the owner's consent (CI green).

What B08 changes for users: Forge's progress bar follows the upscale (phases, DiT blocks; in img2img Batch the
counter moves past each file); Interrupt stops a run within seconds even inside the DiT step or the tiled VAE decode
(hooks on the models' container modules, always removed); Color Correction (6 methods, `lab` default) and Attention
(`sdpa` plus installed backends, stepping down 3 → 2 → `sdpa` when one fails to import) under Advanced Settings, both
in the infotext and as the last two API arguments (21 in all); a GPU Profile (Manual / 8 / 12 / 24 GB) that fills
BlockSwap and the VAE tile size, not a script argument; the title "SeedVR2 Upscaler" — API calls with the old title
get HTTP 422 (README, 🟡 R4 in `07-RISKS.md`).

Live, 8 GB GPU without bf16, 3B GGUF, `--cuda-malloc`: the bar moves 0 → 0.13 → 0.42 (inside the DiT) → 0.6 → done;
Interrupt mid-DiT ends the run in 3.7 s, mid-decode in 0.6 s, VRAM back to 0.8 GB; all six color corrections give
an image; an unavailable `sageattn_2` falls back to `sdpa` with a console note. The owner checked the UI. A
1024×768 output took 190–220 s with `--cuda-malloc` both with and without B08 (decode 2.5–3 min, R6).

The owner's local Forge is on `main`, stopped.

`seedvr2_vendor/` and the `.pt` embeddings match upstream numz/ComfyUI-SeedVR2_VideoUpscaler `4490bd1`, plus the local
patches in `external-apis/seedvr2-upstream.md` (none added in B08). Known problems: `07-RISKS.md` (top: 🟡 R6 VAE
decode spill with `--cuda-malloc`).

Tests: 3115 passed, 1 skipped (POSIX-only), 3 xfailed. Run: `uv run --no-project --python 3.10 --with-requirements
requirements-dev.txt python -m pytest tests -q` (pulls CPU torch on Windows). CI: GitHub Actions on every push.

## Completion percentage

**84%** (28 of 33). 1 block open.

## Next steps

1. B12 — DiT tiling with feather blending (4K+ on 8 GB), then 6K/8K options. Weight 5: plan on a strong model.

Agreed with the owner: REQ-1…REQ-6, simplified Git Flow, ComfyUI files kept as in upstream (Forge Neo only),
hardware target — 8 GB GPU without bf16, the whole repository under Apache-2.0, no grid of upscaled images, the UI
default Scale factor ×2, the title "SeedVR2 Upscaler" without an alias, GPU profiles Manual / 8 / 12 / 24 GB only.
The fork is public: `main` is what users get, pushes only with consent.

## Open questions for the owner

None.
