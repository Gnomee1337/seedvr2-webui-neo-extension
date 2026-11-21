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
6. Controls: the size group — output size mode (Scale factor — default, ×2 / Shortest edge / Long edge) with only
   the chosen mode's control shown: resolution (shortest edge, 512–3840), scale factor (×1.0–×4.0, quick ×1.5 / ×2 / ×4
   buttons), long edge (1024–3840 px) — and the output size line; "After run" (Unload SeedVR2 — default / Keep in
   RAM), unload SD; GPU profile, BlockSwap blocks (0–36), BlockSwap I/O layers (BlockSwap is on when either is set);
   "Advanced Settings": seed, save original, force reload, debug logs, input/latent noise, VAE tiling, tile size and
   overlap, color correction (6 methods, `lab` default), attention (`sdpa` plus the backends whose package is
   installed, looked up without importing). The order on screen is not the order of `run()`'s arguments (that is the
   returned list).
7. Output size line (`seedvr2_neo/size_preview.py`): the size the upscale will give, from the same rules as the run.
   txt2img: from Forge's width, height and Hires. fix controls (found by `elem_id` in `Script.after_component`;
   Hires. fix size by Forge's formula, rounded to `res_step`), updated on any change of them or of the size
   controls, and once after the page loads (`javascript/seedvr2.js` presses a hidden button: Forge applies
   `ui-config.json` after `ui()`, so the line made there may name the code defaults); without those controls (a UI
   layout setting can build the scripts first) a generic note. An image too thin to upscale: "too thin to upscale". img2img: the
   image size is read in the browser (`javascript/seedvr2.js`), updated on a size control change and, through the
   hidden refresh button, within a second of the input image changing; no image or the Batch tab → "depends on each
   input image".
8. GPU profile (`seedvr2_neo/gpu_profiles.py`, `apply_gpu_profile` in the script): Manual (default, not saved in
   `ui-config.json`) / 8 / 12 / 24 GB. Choosing a profile, or changing the DiT while one is chosen, sets BlockSwap (by
   the DiT's size from its header; no listed DiT counts as 3B), VAE tile size and overlap; moving one of those sliders
   by hand (Gradio's `.input`, not the profile's own update) sets the profile back to Manual. Not a `run()` argument.

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
0. Before any generation (process 3, step 1): the numeric settings are read and clamped to the slider ranges
   (`parse_upscale_settings` in `seedvr2_neo/upscale_settings.py`; resolution 512–3840, scale factor 1–4, long edge
   1024–3840, noise 0–1, tile 512–2048, overlap 64–512 and at most half the tile); an unreadable API value takes the
   slider's default, a missing or unknown size mode means Shortest edge (`FALLBACK_SIZE_MODE`; the UI starts in Scale factor). Omitted size arguments (older API callers) → Shortest edge.
   Color correction and attention (`resolve_pipeline_options` in the script, `seedvr2_neo/pipeline_options.py`): a
   missing or unknown value → `lab` / `sdpa`; a backend whose package the vendored pipeline could not import → its
   version 2 if that imported (FlashAttention 3 → 2, SageAttention 3 → 2), else `sdpa`, with a console line.
1. Seed: the UI value, or for `-1` — the first generated image's seed (txt2img), else `p.seed`, else random. One seed
   for all images of the run. A typed seed outside 0 … 2³²−1 (the pipeline also seeds numpy) or not a number counts as
   `-1`, with a console line.
2. Each input → RGB, or RGBA when it has transparency (`to_pipeline_image`: grayscale, palette, CMYK; 16-bit
   grayscale scaled through numpy). This converted image is also what comes back if the upscale fails.
3. Saving plan (`plan_saving` in `seedvr2_neo/run_results.py`): results are saved when "Always save all generated
   images" is on (the request's `override_settings` first, `read_option`) and the caller did not set
   `do_not_save_samples` (the API's `save_images`). Not `p.save_samples()`: Forge's skipped flag from the last txt2img
   batch would stop saving. With saving off nothing is written. "Save Extra Copy of Original" saves the img2img inputs
   with `-original` before the upscale; txt2img base images are saved by Forge.
4. "Unload SD Checkpoint" → `modules.sd_models.unload_model_weights()` + `devices.torch_gc()`.
5. Passes (`UpscaleRun.upscale_images` in `seedvr2_neo/upscale_run.py`; `group_by_shape` + `split_into_passes`): the inputs are grouped by size and mode. Each size gets
   its `OutputTarget` (`UpscaleSettings.target_for`): Shortest edge — the slider value for every size, the long edge
   capped at 4× it; Scale factor — the factor times the input's size, the long edge at most 3840
   (`MAX_OUTPUT_LONG_EDGE`); Long edge — the shortest edge giving the chosen long edge. In the last two the cap is the
   pipeline's `max_resolution`; in Scale factor a shortest edge under 512 is raised towards 512 only while the long edge
   stays within the cap (a very wide panorama keeps the cap). An image so thin that the capped short side is under
   2 px cannot be upscaled: the pipeline fails on it and it comes back with `SeedVR2 Error`. A console line per group
   names the output size (and the raise, with the shortest edge reached). Each group is cut into passes of at most 16 MP of upscaled
   output (`PASS_BUDGET_PIXELS`, counted with that size's output); a larger single frame gets its own pass. Each
   pass → one tensor `[N, H, W, C]` float16 (built per image).
6. Cached models (`GlobalModelCache`, ids `forge_dit` / `forge_vae`) are dropped before the run when "Force Reload" is on
   or when the selected files are not the ones they were loaded from (absolute path, size, modification time, read at
   this step, once per run — `seedvr2_neo/loaded_models.py`).
7. Memory plan from the UI (`seedvr2_neo/memory_plan.py`): "Keep in RAM" → models cached; BlockSwap blocks > 0 or I/O
   layers on → DiT and VAE offload device = CPU (BlockSwap requires it; the VAE waits in RAM during the DiT step).
8. Per pass (`UpscaleRun._upscale_tensor`): a new generation context on Forge's device (`modules.devices.device`), tensors offloaded to CPU.
   `prepare_runner`: DiT/VAE taken from the cache if the file name matches, otherwise loaded from the folder of
   process 2; config `configs_7b` for 36 transformer blocks in the checkpoint header, `configs_3b` for 32 (header
   unreadable → `7b` in the file name, any case). Tiling, BlockSwap and attention from the UI.
   Fixed: no `torch.compile`.
   Compute dtype is bf16 whenever the bf16 probe passes, including GPUs that only emulate it.
9. Phases 1–4 over all frames of the pass, each frame on its own (batch 1, no overlap): encode (VAE) → upscale (DiT,
   one step; text embeddings loaded here) → decode (VAE) → post-process (the chosen color correction). Interrupt is
   checked between passes, between batches (the pipeline's callback and `ctx['interrupt_fn']`) and before every
   container module of the VAE and the DiT (forward pre-hooks set for the four phases and always removed), so a run
   stops within seconds even inside the DiT step or a tiled decode. Progress (`seedvr2_neo/run_progress.py`): before
   the run the script makes the upscale job 1 of 1 unless Forge is inside a job of its own (img2img Batch: after the
   run `job_no` moves past the file's `n_iter` jobs, since `process_images` that would do it is not called), then
   `shared.state.sampling_step` (of 1000) follows the share done — passes by their images, phases by weight, batches,
   and each DiT block — and `shared.state.textinfo` names the phase (and the pass when there are several). After each of the first three phases dropped objects are collected and the
   allocator's cached VRAM is returned (`gc.collect()` + `devices.torch_gc()`). With PyTorch's `cudaMallocAsync`
   allocator (Forge's `--cuda-malloc`), whose pool keeps every freed block, the decode also returns the unused pool
   (`torch.cuda.empty_cache()`) before the decoder, its mid block, each up block and their resnets / upsamplers
   whenever under 3 GB of the GPU is free; the native allocator is left alone. A model that is not cached is dropped
   straight from the GPU, without a copy to RAM (local patch, `external-apis/seedvr2-upstream.md`); with "Unload" the
   next pass loads it again.
10. Output tensor → PIL images one frame at a time (`tensor_to_images`) (N per pass, checked).
11. End of run (`seedvr2_neo/model_lifecycle.py`): "Keep in RAM" after all passes succeeded → models stay cached
   in system RAM (the pipeline moves them off the GPU after their phase); otherwise — "Unload", an error or an
   interrupt — both models are removed from the cache. Either way `gc.collect()` + `devices.torch_gc()`; the console says which.
12. Result (`build_result` in the script, `finish_images` in `seedvr2_neo/run_results.py`): one image and one infotext per input, in input order. The infotext is the
   base one (`BaseInfotext` in `seedvr2_neo/run_results.py`: Forge's infotext of the generated image in txt2img; the
   prompt and `Negative prompt:` line in img2img) with `SeedVR2 Model`, `SeedVR2 VAE`, `SeedVR2 Resolution` (the shortest
   edge this input was upscaled to), `SeedVR2 Seed`, `SeedVR2 Size Mode`, by mode `SeedVR2 Scale` or
   `SeedVR2 Long Edge`, then `SeedVR2 Color Correction` and `SeedVR2 Attention` (the backend used) added to its
   parameters line. Forge's `create_infotext` is not used: it reads the SD model, which is
   unloaded (or never loaded) by then. The infotext is also put into each returned image's `info["parameters"]`
   (with `enable_pnginfo`), so unsaved gallery and API images carry it. An upscaled image is saved to
   `p.outpath_samples` with `-seedvr2` by the saving plan (`samples_format` and `save_to_dirs` from the request's
   `override_settings` first). An image not upscaled (interrupt or error) comes back as its converted input with
   `SeedVR2 Error: Interrupted` / `Runtime Error: <exception type>` added, is not saved, and the result gets a comment
   (`p.comment`, shown under the parameters) saying how many images came back without upscaling.

**Writes:** one `-seedvr2` file per upscaled image when saving is on, plus `-original` files for img2img inputs if
chosen. A failed write (any exception, including other extensions' save callbacks) is printed and skipped. The SD
checkpoint (if unloaded) reloads on the next generation.
**On failure:** interruption or any exception inside the upscale → the console says why (full message and traceback
for errors; the infotext gets only the exception type, since it leaves the machine with the image); the images of
finished passes are returned upscaled (and saved), the rest as their inputs with `SeedVR2 Error`. The models are
unloaded; nothing of the run is reused.
**Why:** `05-DECISIONS.md` → "VRAM and caching".
