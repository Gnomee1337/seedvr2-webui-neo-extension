# PROJECT STATE — seedvr2-webui-neo-extension

## Where we stopped

**B13 closed** and merged into `main` with the owner's consent (CI green). B12 (DiT tiling, 6K / 8K) was deferred by the owner to `06-IDEAS.md`.

What B13 changes for users: with Forge's `--cuda-malloc` (PyTorch `cudaMallocAsync`) the VAE decode returns the
allocator's unused pool before each decoder stage while free VRAM is low (3 GiB at 512 px tiles, scaled by the tile
area), so it no longer spills into shared memory. Automatic, no new setting or script argument; the native allocator
is untouched.

Live, 8 GB GPU without bf16, 3B GGUF Q6_K, tile 512/128, img2img ×2, decode with the flag before → after (without):
1024×768 17.5–27.8 → 9.0 s (8.0 s); 1364×1024 28.8 → 18.3 s (15.6 s); 2048² 112 → 52 s (42 s); no shared-memory
growth; Interrupt mid-decode ends the run in ~1 s, VRAM back to 0.8 GB. Not measured: 7B, batches, other tiles and
GPUs, Linux (`07-RISKS.md` §3).

The owner's local Forge is on `main`, stopped (a local `b13-test` branch exists in its extension clone).

`seedvr2_vendor/` and the `.pt` embeddings match upstream numz/ComfyUI-SeedVR2_VideoUpscaler `4490bd1`, plus the local
patches in `external-apis/seedvr2-upstream.md` (none added in B13). Known problems: `07-RISKS.md` (top: 🟡 R6 the
default ×2 makes UI runs much bigger).

Tests: 3144 passed, 1 skipped (POSIX-only), 3 xfailed. Run: `uv run --no-project --python 3.10 --with-requirements
requirements-dev.txt python -m pytest tests -q` (pulls CPU torch on Windows). CI: GitHub Actions on every push.

## Completion percentage

**100%** (31 of 31). No block open.

## Next steps

1. Optional: `/finish` — final acceptance of the whole project on the strongest model.

Agreed with the owner: REQ-1…REQ-6, simplified Git Flow, ComfyUI files kept as in upstream (Forge Neo only),
hardware target — 8 GB GPU without bf16, the whole repository under Apache-2.0, no grid of upscaled images, the UI
default Scale factor ×2, the title "SeedVR2 Upscaler" without an alias, GPU profiles Manual / 8 / 12 / 24 GB only.
The fork is public: `main` is what users get, pushes only with consent.

## Open questions for the owner

None.
