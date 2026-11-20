# PROJECT STATE — seedvr2-webui-neo-extension

## Where we stopped

**B11 closed** on `feature/b11-output-size` (all commits local; push, CI and merge into `main` wait for the owner's consent).

What B11 changes for users: the output size has three modes — Scale factor (the UI default, ×2, quick ×1.5 / ×2 / ×4),
Shortest edge (the old mode; the fallback for API calls without a mode) and Long edge (1024–3840 px); each input size
gets its own size, the long edge is capped at 3840 in the new modes, a line under the controls shows the final size
before Generate (txt2img from width / height / Hires. fix, img2img read in the browser), and the UI is regrouped (size
group, memory row, rare settings under "Advanced Settings"). The mode is not saved in `ui-config.json`, so the UI always
starts in Scale factor. Heavier default runs: 🟡 R6 in `07-RISKS.md`.

Live, 8 GB GPU without bf16, 3B GGUF: ×2 of 512×384 → 1024×768 in ~50 s; 16:9 → 3840×2160 in ~8 min without
`--cuda-malloc`, ~17.5 min with it (clean); Long edge 1024 on 1920×640 → 1022×340 (clean). The UI start state checked
through the running Forge's UI config, with an old `ui-config.json`.

The owner's local Forge: extension on the local test branch `b11-test` (= this branch), Forge stopped. To return:
`git -C <forge>\extensions\seedvr2-webui-neo-extension checkout main` after the merge, then `merge --ff-only origin/main`.

`seedvr2_vendor/` and the `.pt` embeddings match upstream numz/ComfyUI-SeedVR2_VideoUpscaler `4490bd1`, plus the local
patches in `external-apis/seedvr2-upstream.md`. Known problems: `07-RISKS.md` (top: 🟡 R6 VAE decode spill with
`--cuda-malloc`).

Tests: 2471 passed, 1 skipped (POSIX-only), 3 xfailed. Run: `uv run --no-project --python 3.10 --with-requirements
requirements-dev.txt python -m pytest tests -q` (pulls CPU torch on Windows). CI: GitHub Actions on every push.

## Completion percentage

**75%** (25 of 33). 2 blocks open. B11's weight went 2 → 3 (preview and regrouped UI added at the owner's request);
B08 lost its "simple main UI" item, its weight stays 3.

## Next steps

1. With the owner's consent: push `feature/b11-output-size`, wait for CI, merge into `main` (`--no-ff`).
2. B08 — settings and feedback: color correction and attention mode, Forge progress bar, interrupt inside the DiT
   step, title without "(24G)", GPU profiles (8 / 12 / 24 GB).
3. B12 — DiT tiling with feather blending (4K+ on 8 GB), then 6K/8K options.

Agreed with the owner: REQ-1…REQ-6, simplified Git Flow, ComfyUI files kept as in upstream (Forge Neo only),
hardware target — 8 GB GPU without bf16, the whole repository under Apache-2.0, no grid of upscaled images, the UI
default Scale factor ×2. The fork is public: `main` is what users get, pushes only with consent.

## Open questions for the owner

1. Push the branch and merge B11 into `main`?
