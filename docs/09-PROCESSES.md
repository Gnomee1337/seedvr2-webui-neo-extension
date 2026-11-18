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
`gguf` is not installed: Forge Neo's bundled copy is used (`external-apis/forge-neo.md`).

## 2. Building the UI and finding models  🟢
**Trigger:** Forge builds the txt2img and img2img tabs (startup, "Reload UI"); the 🔄 button; every run.
`Script.ui()` / `Script._refresh_models()` in `scripts/seedvr2_script.py`, `scan_models` in `seedvr2_neo/model_discovery.py`.

1. Model folder: `SeedVR2` (any letter case) inside Forge's `shared.models_path`, if it holds a model file; otherwise
   the same inside `<extension>/models/`; otherwise none. Absolute path, resolved again on every scan.
2. Files ending in `.gguf` / `.safetensors` (exact lower case) are listed, sorted. Names containing `vae` → VAE dropdown,
   the rest → DiT dropdown. Default VAE — the one containing `ema_vae_fp16`.
3. If no folder, no DiT or no VAE is found, a note under the dropdowns names the folder to put files in.
4. 🔄 rescans and updates both dropdowns and the note; a selected file that is still there stays selected.
5. Controls: seed, resolution (shortest edge, 512–3840), unload SD, save original, force reload, debug logs;
   "Advanced": input/latent noise, VAE tiling, tile size and overlap.

**Writes:** nothing.
**On failure:** an unreadable folder counts as empty — the note is shown.

## 3. txt2img + SeedVR2  🟡
**Trigger:** Generate in txt2img with the script selected. `Script.run()`.

0. Models are rescanned (process 2). No folder, nothing selected, or a selected file not in the folder → one line in the
   console and an empty result with `SeedVR2: <reason>`; no base image is generated. Otherwise the folder and file names
   are printed.
1. Remove this script from `p.scripts.scripts`, call `processing.process_images(p)` (normal generation with all
   other extensions, images saved as usual), put the script back.
2. If interrupted — return the base result. Take `processed_base.images[0]` (with batch > 1 and grids enabled this is the grid).
3. "Save Extra Copy of Original" → save that image again with `-original`.
4. Continue with the common upscale (process 5).

**Why:** `05-DECISIONS.md` → "Forge integration".

## 4. img2img + SeedVR2  🟡
**Trigger:** Generate in img2img with the script selected. `Script.run()`.

0. The same model check as in process 3, step 0.
1. The img2img diffusion pass is skipped; `p.init_images[0]` is the input (mask, denoise and resize settings are ignored).
2. Continue with the common upscale (process 5).

## 5. SeedVR2 upscale (common part)  🟡
1. "Unload SD Checkpoint" → `modules.sd_models.unload_model_weights()` + `devices.torch_gc()`.
2. Seed: the UI value, or for `-1` — the generation's first seed, else `p.seed`, else random.
3. Image → tensor `[1, H, W, C]` float16.
4. "Force Reload" → clear `self.runner_cache` (the generation context only; models in `GlobalModelCache` stay).
5. Generation context: reuse `self.runner_cache['ctx']` or create one on `cuda:0`, tensors offloaded to CPU.
6. `prepare_runner`: DiT/VAE taken from `GlobalModelCache` (ids `forge_dit` / `forge_vae`) if the file name matches,
   otherwise loaded from the folder of process 2; config `configs_7b` for 36 transformer blocks in the checkpoint header,
   `configs_3b` for 32 (header unreadable → `7b` in the file name, any case). Tiling from the UI.
   Fixed: no BlockSwap, no DiT/VAE offload device, `sdpa` attention, no `torch.compile`.
   Compute dtype is bf16 whenever the bf16 probe passes, including GPUs that only emulate it.
7. Text embeddings loaded once per context.
8. Phases 1–4: encode (VAE) → upscale (DiT, one step) → decode (VAE) → post-process (LAB color correction).
   Interruption is checked between phases.
9. Output tensor → PIL; `SeedVR2 Model` and `SeedVR2 Resolution` added to the generation parameters;
   saved to `p.outpath_samples` with `-seedvr2`; returned as the only image of the result.

**Writes:** one image file per run (plus `-original` if chosen). After each phase the model is moved to CPU, so
DiT and VAE stay cached in system RAM, not VRAM; the SD checkpoint (if unloaded) reloads on the next generation.
**On failure:** interruption → the input image is returned; any other exception → traceback in the console,
empty result with `Runtime Error: …`. The cached context may keep partial state.
**Why:** `05-DECISIONS.md` → "VRAM and caching".
