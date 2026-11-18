# Ideas in the queue

Only open ideas that have not been taken into work. Nothing starts from here without a separate decision
by the owner. An idea that is taken becomes a block in `docs/03-ROADMAP.md` and leaves this file.

---

## SeedVR2 as a Forge upscaler (Extras tab, Hires. fix)
Now it is only a Script, so it cannot be combined with another Script (XYZ plot, SD upscale) and is absent from Extras.
Options: a postprocessing script for the Extras tab (weight 2, recommended — reuses the same pipeline call;
[wdhameli0/sd-webui-forge-neo-seedvr2](https://github.com/wdhameli0/sd-webui-forge-neo-seedvr2) already does this, Apache-2.0 —
see `external-apis/seedvr2-upstream.md`, "Related project");
registering an `Upscaler` for Hires. fix / SD upscale (weight 4, the upscaler API passes tiles and a scale, not a target size).
Do after: B06, B07. Questions for the owner: which entry point matters more to you?

## Sync `src/` with the latest upstream
`src/` is ComfyUI-SeedVR2_VideoUpscaler v2.5.24. Options: one-off sync with a diff review (weight 3);
keep `src/` as a git subtree with a patch list (weight 4, recommended if syncing often).
Do after: B09 (tests catch regressions). Questions for the owner: how often do you want to follow upstream?

## Automatic model download
`src/utils/downloads.py` and `model_registry.py` already download from Hugging Face with SHA-256 checks.
Options: a "Download" button next to the DiT dropdown (weight 2, recommended); download on first run (weight 1, surprises users with
multi-GB downloads). Questions for the owner: is network access from the extension acceptable?

## Several model folders
Now one folder is used: `<models_path>/SeedVR2`, else the extension's `models/SeedVR2`. Options: list both folders'
models in one dropdown with the folder shown (weight 2); read Forge's extra model paths, if Neo has them (weight 2, needs
a fact check first). Questions for the owner: does anyone keep SeedVR2 models outside one folder?

## Image tiling for very large outputs
VAE tiling cuts VRAM for encode/decode, but the DiT still sees the whole image. Splitting the image into overlapping tiles with
feather blending (as `tiling.py` in the related project) allows 4K+ outputs on 12–16 GB (weight 3). Do after: B06.

## Scale presets (2× / 3× / 4×, long edge)
Now only "shortest edge" in pixels. Presets relative to the input size are easier for users (weight 1). Part of B08 if taken.

## Optional attention backends
SageAttention / FlashAttention are supported in `src/` but fixed to `sdpa`. Expose them in "Advanced" when installed (weight 1). Part of B08 if taken.

## Settings in Forge's Settings tab
Model folder, default unload behaviour and default tile size as Forge options (`shared.opts`) instead of per-run controls (weight 2).

---

## Rejected by the owner
So we don't propose them again. One line each: idea — why not.
