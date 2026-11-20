# Glossary

Terms in plain words. A new term comes up in conversation — add it here right away. Grouped by topic.

## Service and technology
- **CI** — an automatic test run on every push to the repository.
- **Vendored code** — a copy of another project's source kept inside ours (`seedvr2_vendor/`), instead of installing it as a package.
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
- **Package shadowing** — two extensions ship a Python package with the same name (upstream SeedVR2 ports use `src`);
  Python uses whichever was imported first for both. Ours is renamed to `seedvr2_vendor` to avoid it.
- **xfail test** — a test that is expected to fail because it reproduces a known, not yet fixed bug.
- **VAE tiling** — encoding/decoding the image in overlapping tiles to cut VRAM; tile size and overlap are set in the UI.
- **BlockSwap** — keeping part of the DiT blocks in RAM and moving them to the GPU one by one; slower, needs less VRAM.
- **Reserved VRAM** — memory PyTorch keeps after tensors are freed, to reuse it quickly; other programs and the next
  phase cannot use it until the cache is emptied.
- **Shared memory spill** — on Windows, when a program needs more VRAM than the card has, the driver pages the excess
  into system RAM ("shared GPU memory"); no error, but many times slower.
- **`--cuda-malloc` (cudaMallocAsync)** — a Forge launch flag that switches PyTorch to another VRAM allocator; changes
  how much memory stays reserved, and so the speed on small GPUs.
- **Keep in RAM / Unload** — what happens to the SeedVR2 models after a run: kept in system RAM for a faster next run,
  or freed completely (default).
- **Offload** — moving a model or tensors to the CPU when not in use.
- **Input / latent noise scale** — noise added to the input image / to the latent before upscaling; more noise — more invented detail.
- **Color correction** — matching the colors of the result to the input after the upscale, to avoid color shift.
  Methods: `lab` (in LAB color space, the default and closest), `wavelet` / `wavelet_adaptive` (low frequencies of the
  input, details of the result), `hsv`, `adain` (mean and spread of each channel), `none`.
- **Text embeddings (`pos_emb.pt`, `neg_emb.pt`)** — precomputed prompt vectors; SeedVR2 takes no user prompt.
- **sdpa** — PyTorch's built-in scaled dot-product attention; the default attention backend.
- **Attention backend** — the code that computes the DiT's attention: `sdpa`, or FlashAttention / SageAttention when
  their packages are installed (faster on supported GPUs, same model).
- **GPU profile** — a preset that fills BlockSwap and the VAE tile size for an 8, 12 or 24 GB GPU; the sliders stay editable.
- **Forward pre-hook** — a function PyTorch calls before a module (a layer or a block of layers) runs; here it checks
  Interrupt and moves the progress bar inside a long phase.
- **Container module** — a module that holds other modules (a DiT block, a VAE stage), as opposed to a single layer.
- **Output size mode** — how the result size is chosen: Scale factor (×1–×4 of each image, the UI default ×2), Shortest edge (one shortest side in px for every image, the API default) or Long edge (the longest side in px).
- **Size line** — the line under the size controls naming the final output size before Generate; computed with the same rules as the run.
