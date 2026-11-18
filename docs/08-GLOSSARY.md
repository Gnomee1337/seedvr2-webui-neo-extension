# Glossary

Terms in plain words. A new term comes up in conversation — add it here right away. Grouped by topic.

## Service and technology
- **CI** — an automatic test run on every push to the repository.
- **Vendored code** — a copy of another project's source kept inside ours (`src/`), instead of installing it as a package.
- **Upstream** — the project our vendored code was copied from (ComfyUI-SeedVR2_VideoUpscaler).
- **Stub** — a fake replacement of a module (here: Forge's `modules.*`) so code can be tested without running Forge.

## Forge Neo
- **Forge Neo** — the `neo` branch of `Haoming02/sd-webui-forge-classic`, a fork of Stable Diffusion WebUI Forge.
- **Extension** — a folder in `extensions/`; Forge runs its `install.py` at launch and loads `scripts/*.py`.
- **Script (selectable)** — an entry in the "Script" dropdown at the bottom of txt2img/img2img; when chosen, its `run()` replaces normal generation.
- **AlwaysOn script** — a script that runs on every generation alongside others (not used here).
- **`p` (processing object)** — Forge's object with all generation parameters (`StableDiffusionProcessingTxt2Img` / `…Img2Img`).
- **`Processed`** — what a generation returns to the UI: images, seeds, infotexts.
- **Infotext** — the generation parameters text stored in the PNG and shown under the gallery; used by "Send to" and "PNG Info".
- **Grid** — a combined image of all batch images; Forge puts it first in the results when "return grid" is on.
- **Interrupt / Stop generating** — Forge's two stop buttons: abort now / finish the current image and stop.

## SeedVR2
- **SeedVR2** — ByteDance's one-step diffusion model for image and video restoration / upscaling.
- **DiT** — Diffusion Transformer, the main SeedVR2 network (3B or 7B parameters).
- **VAE** — the encoder/decoder between pixels and latents (`ema_vae_fp16.safetensors`).
- **Latent** — the compressed representation the DiT works on.
- **3B / 7B** — model sizes in billions of parameters; 7B is better and much heavier.
- **"sharp" variant** — a 7B checkpoint tuned for more detail.
- **GGUF, Q4_K_M, Q8_0** — a file format for quantized (compressed) weights and its compression levels; less VRAM, slightly lower quality.
- **fp16 / bf16 / fp8** — number formats of weights; fp8 halves memory against fp16.
- **Checkpoint header** — the start of a model file that lists tensor names and shapes; read without loading weights.
  The number of transformer blocks in it (32 or 36) tells 3B from 7B.
- **Package shadowing** — two extensions ship a Python package with the same name (here `src`); Python uses whichever
  was imported first for both.
- **xfail test** — a test that is expected to fail because it reproduces a known, not yet fixed bug.
- **VAE tiling** — encoding/decoding the image in overlapping tiles to cut VRAM; tile size and overlap are set in the UI.
- **BlockSwap** — keeping part of the DiT blocks in RAM and moving them to the GPU one by one; slower, needs less VRAM.
- **Offload** — moving a model or tensors to the CPU when not in use.
- **Input / latent noise scale** — noise added to the input image / to the latent before upscaling; more noise — more invented detail.
- **LAB color correction** — matching the colors of the result to the input in LAB color space, to avoid color shift.
- **Text embeddings (`pos_emb.pt`, `neg_emb.pt`)** — precomputed prompt vectors; SeedVR2 takes no user prompt.
- **sdpa** — PyTorch's built-in scaled dot-product attention; the default attention backend.
