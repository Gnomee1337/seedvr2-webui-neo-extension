# SeedVR2 upscaler for Stable Diffusion WebUI Forge Neo

An extension for [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) that upscales
images with [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) (ByteDance's one-step diffusion restoration
model). It runs as a **Script** in txt2img and img2img. The SeedVR2 pipeline is a vendored copy of the ComfyUI
port [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) v2.5.24.

## Installation

1. Forge Neo → **Extensions** → **Install from URL** → paste this repository's URL → **Install**, then restart
   the WebUI. At launch Forge runs `install.py`, which installs `rotary-embedding-torch`.
2. If `install.py` did not run correctly, install the dependencies into the WebUI's Python environment by hand:

   ```
   pip install rotary-embedding-torch gguf
   ```

   `gguf` is needed only for `.gguf` models; Forge Neo does not install it as a separate package.
3. Updating: Forge's **Check for updates**, then **Apply and restart UI**, works as usual. If you cloned the extension by hand and `git pull`
   stops with "local changes would be overwritten" (old versions kept compiled `.pyc` files in git), run
   `git reset --hard origin/main` in the extension folder; it discards local edits to the extension's files.

Install only one SeedVR2 extension for Forge. The original one shows the same name in the **Script** list, and each
keeps its own copy of the models in memory.

## Models

Put the models into `<WebUI root>/models/SeedVR2/` (the folder name may be in any letter case; with
`--model-ref` / `--data-dir` it is `SeedVR2/` inside your models folder). If that folder does not hold both a DiT and
a VAE but `extensions/<this extension>/models/SeedVR2/` does, the extension folder is used instead. Earlier versions looked in the extension folder
first: if you keep full sets in both places, the WebUI folder is now the one used. The folder in use is printed in the
console on every run; when no models are found, the script panel shows the exact path to put them in.

You need:

- a VAE: `ema_vae_fp16.safetensors` (any file with `vae` in its name is listed as a VAE);
- a DiT model, for example `seedvr2_ema_7b_sharp-Q4_K_M.gguf`, `seedvr2_ema_3b_fp16.safetensors`.
  The file can be renamed freely: 3B or 7B is read from the file itself.

Downloads: [numz/SeedVR2_comfyUI](https://huggingface.co/numz/SeedVR2_comfyUI) and
[AInVFX/SeedVR2_comfyUI](https://huggingface.co/AInVFX/SeedVR2_comfyUI) on Hugging Face.
After adding a model, press 🔄 next to the model dropdowns — no restart needed.

## Usage

Select **SeedVR2 Native Upscaler (24G)** in the **Script** dropdown at the bottom of the txt2img or img2img tab.
Usually you only need to change **Upscale Resolution (Shortest Edge)**.

- **txt2img** — the images are generated as usual (with all other extensions), then every image of the batch is
  upscaled with SeedVR2 (the grid is not upscaled, and no grid of the results is made).
- **img2img** — the img2img diffusion pass is skipped; every input image is upscaled with SeedVR2 directly (the
  **Batch** tab works too). Grayscale, palette and 16-bit images become RGB; transparency is kept.

Each result keeps its generation parameters plus `SeedVR2 …` keys (model, VAE, resolution, seed). Files are written
like Forge's own: with **Settings → Saving images → Always save all generated images** on, each upscaled image is saved
with the `-seedvr2` suffix; with it off (or `save_images: false` in an API call) the extension writes no files at all.
**Save Extra Copy of Original** also saves the img2img input image (`-original`); txt2img images are saved by Forge.
If the upscale is interrupted or fails, the images come back as they were, with `SeedVR2 Error` in their parameters
and a note under the result (API callers: look for `SeedVR2 Error` in `info` → `infotexts`). **Unload SD Checkpoint** (on by default) frees VRAM for SeedVR2; the checkpoint is
reloaded on the next generation.

**VAE Tile Size** (Advanced Settings) defaults to 512 and **Tile Overlap** to 128; the overlap used is at most half
the tile size. The slider was called "Tile Size" with a default of 1024 until November 2026: after updating from such a
version it is reset to 512 one time, so a larger value set on purpose (on a 12–24 GB GPU it runs fewer tiles) needs
setting again.

**After run:** *Unload SeedVR2* (default) frees the SeedVR2 models from VRAM and RAM when the upscale ends;
*Keep in RAM* keeps them for a faster next run at the cost of system RAM about the size of the two model files
(3–4 GB for 3B fp8, ~16 GB for 7B fp16). An error or an interrupt during the upscale unloads them in either mode.
**Force Reload** reads the model files from disk again; it matters only with *Keep in RAM*, since *Unload* reads
them on every run anyway.

### GPUs with 8–11 GB

- **7B models need BlockSwap.** Without it they spill into shared memory and run many times slower. Set
  **BlockSwap: DiT blocks kept in RAM** to **16** (the script shows this hint when a 7B is selected). Measured with
  7B fp8 at 1080px on an 8 GB GPU: the DiT step went from ~13 minutes to 35–42 seconds. Fewer blocks are faster only
  while everything still fits; more blocks use less VRAM and run slower. 3B models need no BlockSwap.
- **VAE Tile Size** (Advanced Settings) is 512 by default. Without `--cuda-malloc` that keeps VAE encoding and
  decoding inside VRAM; with 1024 a 1080px image overflowed into shared memory, and with `--cuda-malloc` a run could
  hang for many minutes. With `--cuda-malloc` the decode at 512 still spills somewhat (next point).
- **`--cuda-malloc`** in Forge's launch options changes the speed: in the same test, VAE decoding took 12 s without
  it and 25–137 s with it (it reserves more VRAM than the card has), while the 7B DiT step with 8 BlockSwap blocks was
  faster with it (21 s vs 122 s). With 16 blocks both settings are close. If VAE decoding is slow, try launching without it.

<img width="1660" height="694" alt="Extension UI" src="https://github.com/user-attachments/assets/777c34e7-aca6-4e51-9994-f02f817311ea" />

## Tests

Tests run outside Forge and need no GPU:

```
pip install -r requirements-dev.txt
python -m pytest tests
```

## Documentation

All documentation is in [`docs/`](docs/README.md). Where the project is now — [`docs/04-STATE.md`](docs/04-STATE.md).
The entry point for Claude Code is [`CLAUDE.md`](CLAUDE.md).

## Credits

- [SeedVR / SeedVR2](https://github.com/ByteDance-Seed/SeedVR) — ByteDance Seed.
- [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) — the pipeline code in `seedvr2_vendor/` (upstream folder `src/`).
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) — the GGUF dequantization code.
- Original Forge extension — akoyaki ayagi.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). Attribution and the list of changes to the vendored code — [`NOTICE`](NOTICE).
