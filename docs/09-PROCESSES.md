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

1. Model folder. Candidates in order: every letter-case spelling of `SeedVR2` inside Forge's `shared.models_path`
   (exact spelling first), then the same inside `<extension>/models/`. The first candidate with both a DiT and a VAE is
   used; if none is complete, the first with any model file; otherwise none. Absolute path with the folder's real
   spelling, resolved again on every scan.
2. Files ending in `.gguf` / `.safetensors` (exact lower case) are listed, sorted. Names containing `vae` → VAE dropdown,
   the rest → DiT dropdown. Default VAE — the one containing `ema_vae_fp16`.
3. If no folder, no DiT or no VAE is found, a note under the dropdowns names the folder to put files in.
4. 🔄 rescans and updates both dropdowns and the note; a selected file that is still there stays selected.
5. Selecting a DiT reads its size from the header (process 5, step 6); a 7B on a CUDA GPU with less than 11 GiB total
   memory shows a note recommending BlockSwap. Unknown size or GPU → no note.
6. Controls: seed, resolution (shortest edge, 512–3840), unload SD, save original, force reload, debug logs;
   "After run" (Unload SeedVR2 — default / Keep in RAM), BlockSwap blocks (0–36), BlockSwap I/O layers (BlockSwap is on when either is set);
   "Advanced": input/latent noise, VAE tiling, tile size and overlap.

**Writes:** nothing.
**On failure:** an unreadable folder counts as empty — the note is shown.

## 3. txt2img + SeedVR2  🟡
**Trigger:** Generate in txt2img with the script selected. `Script.run()`.

0. Models are rescanned (process 2). No folder, no DiT or VAE in it, nothing selected, a selected file not in the folder
   → one line in the console and an empty result with `SeedVR2: <reason>` (also as its infotext); no base image is generated. Otherwise the folder and file names are printed.
   Then the vendored pipeline is imported; an `ImportError` → console line and an empty result with `Error: …`, also
   before any generation or SD unload.
1. Remove this script from `p.scripts.scripts`, call `processing.process_images(p)` (normal generation with all
   other extensions, images saved as usual), put the script back.
2. If interrupted — return the base result as is. The inputs are the generated images: `processed_base.images` from
   `index_of_first_image` (skips the grid), as many as `p.all_seeds` (`select_generated_images` in
   `seedvr2_neo/run_inputs.py`). No upscaled grid is made.
3. No generated image → console line and an empty result `SeedVR2: there is no image to upscale`.
4. Continue with the common upscale (process 5).

**Why:** `05-DECISIONS.md` → "Forge integration".

## 4. img2img + SeedVR2  🟡
**Trigger:** Generate in img2img with the script selected (also once per file of the Batch tab). `Script.run()`.

0. The same model check and pipeline import as in process 3, step 0.
1. The img2img diffusion pass is skipped; every distinct object of `p.init_images` is an input (the Batch tab passes one
   file as `[image] * batch_size`; mask, denoise and resize settings are ignored). None → the empty result of process 3, step 3.
2. Each input gets the SeedVR2 seed and the UI prompt for its file name and infotext; `p` is not changed (the Batch
   tab reuses it between files).
3. Continue with the common upscale (process 5).

## 5. SeedVR2 upscale (common part)  🟡
1. Seed: the UI value, or for `-1` — the first generated image's seed (txt2img), else `p.seed`, else random. One seed
   for all images of the run. An API value that is not a number counts as `-1`.
2. Saving plan (`plan_saving` in `seedvr2_neo/run_results.py`): results are saved when `p.save_samples()` is true
   ("Always save all generated images", the API's `save_images`); "Save Extra Copy of Original" saves the inputs with
   `-original` before the upscale only when Forge did not save them (txt2img with saving on — never) and the caller
   did not disable saving.
3. "Unload SD Checkpoint" → `modules.sd_models.unload_model_weights()` + `devices.torch_gc()`.
4. Each input → RGB, or RGBA when it has transparency (grayscale, palette, 16-bit, CMYK; `to_pipeline_image`). The
   inputs are grouped by size and mode; each group → one tensor `[N, H, W, C]` float16, one pipeline pass.
5. Cached models (`GlobalModelCache`, ids `forge_dit` / `forge_vae`) are dropped before the run when "Force Reload" is on
   or when the selected files are not the ones they were loaded from (absolute path, size, modification time, read at
   this step, once per run — `seedvr2_neo/loaded_models.py`).
6. Memory plan from the UI (`seedvr2_neo/memory_plan.py`): "Keep in RAM" → models cached; BlockSwap blocks > 0 or I/O
   layers on → DiT and VAE offload device = CPU (BlockSwap requires it; the VAE waits in RAM during the DiT step).
7. Per group: a new generation context on Forge's device (`modules.devices.device`), tensors offloaded to CPU.
   `prepare_runner`: DiT/VAE taken from the cache if the file name matches, otherwise loaded from the folder of
   process 2; config `configs_7b` for 36 transformer blocks in the checkpoint header, `configs_3b` for 32 (header
   unreadable → `7b` in the file name, any case). Tiling and BlockSwap from the UI.
   Fixed: `sdpa` attention, no `torch.compile`.
   Compute dtype is bf16 whenever the bf16 probe passes, including GPUs that only emulate it.
8. Phases 1–4 over all frames of the group, each frame on its own (batch 1, no overlap): encode (VAE) → upscale (DiT,
   one step; text embeddings loaded here) → decode (VAE) → post-process (LAB color correction). Interruption is checked
   between phases and between groups. After each of the first three phases dropped objects are collected and the
   allocator's cached VRAM is returned (`gc.collect()` + `devices.torch_gc()`). A model that is not cached is dropped
   straight from the GPU, without a copy to RAM (local patch, `external-apis/seedvr2-upstream.md`); with "Unload" the
   next group loads it again.
9. Output tensor → PIL images (N per group, checked).
10. End of run (`seedvr2_neo/model_lifecycle.py`): "Keep in RAM" after all groups succeeded → models stay cached
   in system RAM (the pipeline moves them off the GPU after their phase); otherwise — "Unload", an error or an
   interrupt — both models are removed from the cache. Either way `gc.collect()` + `devices.torch_gc()`; the console says which.
11. Result (`build_result` in the script): one image and one infotext per input, in input order. The infotext is the
   base one (`BaseInfotext` in `seedvr2_neo/run_results.py`: Forge's infotext of the generated image in txt2img; the
   prompt and `Negative prompt:` line in img2img) with `SeedVR2 Model`, `SeedVR2 VAE`, `SeedVR2 Resolution`,
   `SeedVR2 Seed` added to its parameters line. Forge's `create_infotext` is not used: it reads the SD model, which is
   unloaded (or never loaded) by then. An upscaled image is saved to `p.outpath_samples` with `-seedvr2` by the saving
   plan. An image not upscaled (interrupt or error) comes back as its input with `SeedVR2 Error: <reason>` added and is not saved.

**Writes:** one `-seedvr2` file per upscaled image when saving is on, plus `-original` files by the saving plan. A
failed write is printed and skipped. The SD checkpoint (if unloaded) reloads on the next generation.
**On failure:** interruption or any exception inside the upscale → the console says why (traceback for errors); the
images of finished groups are returned upscaled, the rest as their inputs with `SeedVR2 Error`. The models are
unloaded; nothing of the run is reused.
**Why:** `05-DECISIONS.md` → "VRAM and caching".
