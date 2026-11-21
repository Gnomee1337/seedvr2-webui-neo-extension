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

Install only one SeedVR2 extension for Forge: each keeps its own copy of the models in memory.

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

Select **SeedVR2 Upscaler** in the **Script** dropdown at the bottom of the txt2img or img2img tab.
Usually you only need to choose the output size with **Output Size Mode**:

- **Scale factor** (default, ×2) — ×1.0 to ×4.0 of each image's own size (buttons ×1.5 / ×2 / ×4 next to the
  slider). In txt2img it applies to the final image, after Hires. fix.
- **Shortest edge** — **Upscale Resolution (Shortest Edge)** in px (1080 by default) for every image; the mode of
  earlier versions.
- **Long edge** — the longest side of each image in px (1024–3840; 2048 is "2K", 3840 is "4K UHD" for 16:9).

In the Scale factor and Long edge modes the long edge never goes above 3840 px (larger sizes need a DiT that works in
tiles, which is not there yet) or above the chosen long edge. In Scale factor a small image is raised to 512 px on its
shortest edge, so ×1 of a small image is still an upscale; a very wide panorama is raised only as far as its long edge
stays within 3840 px. Images of different sizes in one img2img batch each get their own size.

The line under the size controls shows the final size before you press Generate: in txt2img from the width, height
and Hires. fix settings, in img2img from the image you loaded (the Batch tab says it depends on each image). The
console also prints it per group of same-sized images, e.g. `[SeedVR2] 2 images 512x384 -> 1024x768`.

- **txt2img** — the images are generated as usual (with all other extensions), then every image of the batch is
  upscaled with SeedVR2 (the grid is not upscaled, and no grid of the results is made).
- **img2img** — the img2img diffusion pass is skipped; every input image is upscaled with SeedVR2 directly (the
  **Batch** tab works too). Grayscale, palette and 16-bit images become RGB; transparency is kept.

While the upscale runs, Forge's progress bar shows its phase (Encoding, Upscaling, Decoding, Post-processing) and
moves inside the long DiT step too. **Interrupt** stops the upscale within a few seconds, even in the middle of the DiT
step or of the VAE decoding (the console may show an `Error in Phase …` line from the pipeline right before
"Process Interrupted by User": that is the interrupt, not a crash); **Skip** and "Stop generating" let it finish.

Each result keeps its generation parameters plus `SeedVR2 …` keys (model, VAE, the shortest edge it was upscaled to,
seed, size mode with its factor or long edge, color correction, attention). Files are written
like Forge's own: with **Settings → Saving images → Always save all generated images** on, each upscaled image is saved
with the `-seedvr2` suffix; with it off (or `save_images: false` in an API call) the extension writes no files at all.
**Save Extra Copy of Original** also saves the img2img input image (`-original`); txt2img images are saved by Forge.
If the upscale is interrupted or fails, the images come back as they were, with `SeedVR2 Error` in their parameters
and a note under the result (API callers: look for `SeedVR2 Error` in `info` → `infotexts`). **Unload SD Checkpoint** (on by default) frees VRAM for SeedVR2; the checkpoint is
reloaded on the next generation.

**VAE Tile Size** (Advanced Settings) defaults to 512 and **Tile Overlap** to 128; the overlap used is at most half
the tile size. Older versions had a slider called "Tile Size" with a default of 1024: after updating from one, the tile size is
reset to 512 one time, so a larger value you had set on purpose needs setting again (on 12–24 GB GPUs a larger tile
means fewer tiles and a faster VAE).

**Color Correction** (Advanced Settings) matches the result's colors to the input: `lab` (default, the closest),
`wavelet`, `wavelet_adaptive`, `hsv`, `adain`, or `none` for the model's own colors. **Attention** is `sdpa` (built into
PyTorch); `flash_attn_2` / `flash_attn_3` / `sageattn_2` / `sageattn_3` appear when their package is installed in
Forge's Python. When the chosen one fails to load, a run uses version 2 of the same backend if that loaded, else
`sdpa`, with a console note; the image parameters name the backend actually used.

**GPU Profile** (next to BlockSwap) fills BlockSwap and the VAE tile size for an 8, 12 or 24 GB GPU and for the selected
DiT (3B or 7B); the sliders stay as you can see them and can be changed afterwards (the profile then shows *Manual*).
The 8 GB values are measured; the 12 and 24 GB ones are a reasonable start, not measured yet.

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
  hang for many minutes.
- **Large outputs take minutes.** The default ×2 of a 1024×1024 SDXL image is 2048×2048, about 3.5× the pixels of the
  1080 px shortest edge earlier versions used; with Hires. fix before it, more. 3B GGUF, VAE tile 512, 8 GB GPU
  without bf16: ×2 of a 512×384 image (1024×768) ~50 s; a 16:9 image to 3840×2160 (Long edge 3840) ~8 minutes without `--cuda-malloc` (DiT step 5.7 min, decoding
  1.5 min); with it ~17.5 minutes (DiT step 10.5 min), measured before the decoding fix below and not re-measured.
  VRAM full in both cases; the image is clean.
  For quick results stay at 2048 or below.
- **`--cuda-malloc`** in Forge's launch options: its allocator never gives freed VRAM back on its own, so during VAE
  decoding the extension returns the unused part between decoder stages whenever free VRAM runs low (under 3 GB with
  the default 512 tiles, more with larger tiles).
  On an 8 GB GPU decoding a 2048×2048 output takes ~52 s with the flag and ~42 s without it (it took ~112 s with the
  flag before, spilling into shared memory). The 7B DiT step with 8 BlockSwap blocks is faster with the flag (21 s vs
  122 s); with 16 blocks both settings are close. With 3B models the flag brings nothing and decodes ~24% slower.

### API

Send `"script_name": "SeedVR2 Upscaler"` to `/sdapi/v1/txt2img` or `/sdapi/v1/img2img` with
`script_args` in this order:

```
dit_model, vae_model, seed, resolution, input_noise, latent_noise, force_reload, unload_sd, save_original,
use_tile_vae, tile_size, tile_overlap, debug_mode, after_run, blocks_to_swap, swap_io_components,
size_mode, scale_factor, long_edge, color_correction, attention_mode
```

`size_mode` is `"Scale factor"`, `"Shortest edge"` or `"Long edge"`. A missing or unknown `size_mode` means
`"Shortest edge"`, as in versions before the modes, although the UI and `/sdapi/v1/script-info` show "Scale factor" as
the default. `color_correction` and `attention_mode` take the strings listed above; a missing or unknown value means
`"lab"` / `"sdpa"`. Forge does not fill omitted arguments with the UI's values, so send all 21 (older callers' 13, 16
or 19 still work: 13 and 16 get the shortest edge, all of them `lab` and `sdpa`). Numbers outside a slider's range are
clamped to it. The GPU profile is not an argument: send the slider values themselves.

Versions before this one were called **SeedVR2 Native Upscaler (24G)**: an API call with that `script_name` now fails
(HTTP 422) — use the new name. In the UI the Script dropdown starts at *None* once after the update; the saved slider
values are kept.

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
