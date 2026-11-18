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

- **txt2img** — the image is generated as usual (with all other extensions), then upscaled with SeedVR2.
- **img2img** — the img2img diffusion pass is skipped; the input image is upscaled with SeedVR2 directly.

The result is saved with the `-seedvr2` suffix. **Unload SD Checkpoint** (on by default) frees VRAM for SeedVR2;
the checkpoint is reloaded on the next generation.

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
