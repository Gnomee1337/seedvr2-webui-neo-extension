# How the system works — process by process

For humans and for Claude: trigger → steps → what changes → settings.
Modules — `02-ARCHITECTURE.md`; why — `05-DECISIONS.md` (the section is given for each process).
The source of truth is the code.

**Status:** 🟢 working · 🟡 works with known problems (see `07-RISKS.md`) · ⚪ disabled.

**Idempotency:** not applicable — every run is started by the user and writes a new image file.

## 1. Installation  🟡
**Trigger:** Forge Neo launch (every start).
`install.py`.

1. If `rotary-embedding-torch` is not installed — `pip install rotary-embedding-torch`.

**Writes:** packages into the WebUI's Python environment.
**On failure:** Forge prints the error and continues; the extension fails later on import (`ImportError` → empty result with an error message).
**Known gap:** `gguf` is not installed, so `.gguf` models fail (`07-RISKS.md` §2).

## 2. Building the UI  🟡
**Trigger:** Forge builds the txt2img and img2img tabs (startup, "Reload UI").
`Script.ui()` in `scripts/seedvr2_script.py`.

1. At import, `MODEL_DIR` = `extensions/<ext>/models/SeedVR2`; if it does not exist — `<webui>/models/SeedVR2` if that exists.
2. Files ending in `.gguf` / `.safetensors` are listed. Names containing `vae` → VAE dropdown, the rest → DiT dropdown;
   if either list is empty, it gets all files. Default VAE — the one containing `ema_vae_fp16`.
3. Controls: seed, resolution (shortest edge, 512–3840), unload SD, save original, force reload, debug logs;
   "Advanced": input/latent noise, VAE tiling, tile size and overlap.

**Writes:** nothing.
**On failure:** no folder → empty dropdowns, no message. New files appear only after a UI restart.

## 3. txt2img + SeedVR2  🟡
**Trigger:** Generate in txt2img with the script selected. `Script.run()`.

1. Remove this script from `p.scripts.scripts`, call `processing.process_images(p)` (normal generation with all
   other extensions, images saved as usual), put the script back.
2. If interrupted — return the base result. Take `processed_base.images[0]` (with batch > 1 and grids enabled this is the grid).
3. "Save Extra Copy of Original" → save that image again with `-original`.
4. Continue with the common upscale (process 5).

**Why:** `05-DECISIONS.md` → "Forge integration".

## 4. img2img + SeedVR2  🟡
**Trigger:** Generate in img2img with the script selected. `Script.run()`.

1. The img2img diffusion pass is skipped; `p.init_images[0]` is the input (mask, denoise and resize settings are ignored).
2. Continue with the common upscale (process 5).

## 5. SeedVR2 upscale (common part)  🟡
1. "Unload SD Checkpoint" → `modules.sd_models.unload_model_weights()` + `devices.torch_gc()`.
2. Seed: the UI value, or for `-1` — the generation's first seed, else `p.seed`, else random.
3. Image → tensor `[1, H, W, C]` float16.
4. "Force Reload" → clear `self.runner_cache` (the generation context only; models in `GlobalModelCache` stay).
5. Generation context: reuse `self.runner_cache['ctx']` or create one on `cuda:0`, tensors offloaded to CPU.
6. `prepare_runner`: DiT/VAE taken from `GlobalModelCache` (ids `forge_dit` / `forge_vae`) if the file name matches,
   otherwise loaded; config `configs_7b` if the DiT file name contains `7b`, else `configs_3b`. Tiling from the UI.
   Fixed: no BlockSwap, no offload, `sdpa` attention, no `torch.compile`.
7. Text embeddings loaded once per context.
8. Phases 1–4: encode (VAE) → upscale (DiT, one step) → decode (VAE) → post-process (LAB color correction).
   Interruption is checked between phases.
9. Output tensor → PIL; `SeedVR2 Model` and `SeedVR2 Resolution` added to the generation parameters;
   saved to `p.outpath_samples` with `-seedvr2`; returned as the only image of the result.

**Writes:** one image file per run (plus `-original` if chosen); the models stay on the GPU.
**On failure:** interruption → the input image is returned; any other exception → traceback in the console,
empty result with `Runtime Error: …`. The cached context may keep partial state.
**Why:** `05-DECISIONS.md` → "VRAM and caching".
