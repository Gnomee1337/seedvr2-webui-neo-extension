# Architecture

WHAT exists in the code: stack, folders, modules, entry points, data. How processes
work step by step — `docs/09-PROCESSES.md`; code rules — `docs/process/code-style.md`; why — `05-DECISIONS.md`.
The source of truth is the code; this document is reconciled with it on every `/block-done`.

## Stack
| What | With |
|---|---|
| Language | Python (the version of the user's Forge Neo, 3.10+) |
| Host | Forge Neo (`Haoming02/sd-webui-forge-classic`, branch `neo`) — the extension runs inside its process |
| UI | Gradio components of the host |
| ML | PyTorch, torchvision, einops, diffusers, OmegaConf, `rotary-embedding-torch`, `gguf` (GGUF models only; Forge Neo's bundled copy), OpenCV (alpha upscaling) |
| Database | none |
| Packages | `install.py` (run by Forge at launch); `requirements.txt` is informational — Forge does not read it |
| Runtime | inside Forge Neo, on Forge's device (`modules.devices.device`); the vendored bf16 probe still uses `cuda:0` |
| Tests | `pytest` outside Forge, no GPU (`requirements-dev.txt`: `pytest`, `gguf`, `numpy`, `pillow`, CPU `torch` + `torchvision`); config in `tests/pytest.ini`; CI — `.github/workflows/tests.yml` (pytest on every push and pull request, Python 3.10, CPU torch) |

## Folders
```
<repo root>/                — the extension root (Forge clones it into extensions/)
  scripts/                  — Forge entry point: seedvr2_script.py
  seedvr2_neo/              — our helpers without Forge/Gradio imports (models, memory, inputs, the pipeline run, results)
  tests/                    — pytest, mirrors scripts/, seedvr2_neo/ and seedvr2_vendor/; pytest.ini lives here
  .github/workflows/        — CI: tests.yml
  seedvr2_vendor/           — vendored SeedVR2 pipeline (ComfyUI-SeedVR2_VideoUpscaler v2.5.24; upstream folder `src/`)
    common/                 — config loading, diffusion samplers/schedules/timesteps, seed, cache, distributed stubs
    core/                   — generation phases, runner/model configuration, model loading, global model cache, inference
    data/image/transforms/  — resize/crop transforms
    interfaces/             — ComfyUI nodes (dead code in Forge)
    models/                 — DiT 3B, DiT 7B, video VAE v3 (+ its YAML config)
    optimization/           — memory manager, BlockSwap, GGUF ops/dequant, compatibility shims
    utils/                  — color correction, constants and model lookup, debug logger, downloads, model registry
  configs_3b/, configs_7b/  — main.yaml per model size
  pos_emb.pt, neg_emb.pt    — precomputed positive/negative text embeddings
  install.py                — installs pip packages at Forge launch
  requirements.txt          — informational dependency list (Forge does not read it)
  LICENSE, NOTICE           — Apache-2.0 text; attribution of vendored code and the list of our changes to it
  __init__.py               — ComfyUI entry point (dead code in Forge)
  docs/                     — documentation
```

## Stack conventions
| What | How in this project |
|---|---|
| All tests / one area | `python -m pytest tests` / `python -m pytest tests/seedvr2_neo` (from the repository root) |
| Test layout | `tests/` mirrors `scripts/`, `seedvr2_neo/` and `seedvr2_vendor/`; no `__init__.py` in test folders, default import mode, so test file names must be unique. `tests/scripts/`: stand-ins for Gradio, Forge's `modules.*` and the vendored pipeline modules in `forge_stubs.py` (shapes from `external-apis/forge-neo.md`), put into `sys.modules` by `conftest.py`, which imports the script afresh for every test |
| Test file / stress scenarios file name | `test_<module>.py` / `test_<module>_stress.py` |
| Linter, formatter | none (CI runs only the tests) |

## Entry points

### Called by Forge Neo
| Hook | Caller | Module | What it does |
|---|---|---|---|
| `install.py` (run as a script) | Forge launcher, every start | `install.py` | `pip install rotary-embedding-torch` if missing |
| `Script.title / show / ui` | Forge, when building the txt2img and img2img tabs | `scripts/seedvr2_script.py` | Adds "SeedVR2 Upscaler" to the Script dropdown; scans the model folder for the DiT/VAE dropdowns, 🔄 rescans it |
| `Script.run(p, *args)` | Forge, when Generate is pressed with this script selected | `scripts/seedvr2_script.py` | txt2img: generate then upscale; img2img: upscale the input image. Returns `Processed` |

### Pages and API for people
None of its own: the UI lives inside the txt2img / img2img tabs.

## Modules

### scripts/
- `seedvr2_script.py` — the Forge side: UI controls, the model refresh button and the 7B low-VRAM note, the model
  check and pipeline import before a run, collecting the images to upscale (txt2img base images / img2img inputs),
  the SD unload, the process-wide `MODEL_LIFECYCLE`, Forge's hooks for the run (`forge_hooks`: device, memory release,
  `shared.state.interrupted`, the progress bar through `start_forge_progress` / `report_forge_progress`), the run's
  color correction and attention (`resolve_pipeline_options`), saving (`-seedvr2`, `-original`) and Forge's
  `Processed`. `ui()` builds four sections (`_model_controls`, `_size_controls`, `_memory_controls`,
  `_advanced_controls`) and returns the controls in `run()`'s order; the GPU profile dropdown is not among them, it
  fills the memory sliders through `apply_gpu_profile` (`_connect_gpu_profile`). The output size preview: `after_component` keeps Forge's txt2img width / height / Hires. fix controls (by
  `elem_id`), whose changes update the size line; in img2img the browser reads the image size.

### javascript/, style.css
Loaded by Forge into the page (any extension's `javascript/*.js` and `style.css`).
- `javascript/seedvr2.js` — on page load presses the hidden txt2img refresh button once (the saved UI values are in
  place by then); the img2img input image size read from the page (`seedvr2Img2imgPreview`, the `_js` of the preview
  events), and a once-a-second check that presses the hidden img2img refresh button when that size changes.
- `style.css` — the hidden refresh button, the size group, the quick scale buttons, the highlighted size line.

### seedvr2_neo/
No Forge or Gradio imports; torch only in `upscale_run.py`. Tested outside Forge.
- `model_discovery.py` — `scan_models(models_path, extension_root)` → `ModelScan` (absolute folder, DiT and VAE file
  names, defaults); the user-facing note for missing models; the reason a selection cannot run.
- `memory_plan.py` — `build_memory_plan(after_run, blocks_to_swap, swap_io_components)` → `MemoryPlan` (cache flag,
  DiT/VAE offload devices, `block_swap_config()`); `AfterRun` (UI labels); `is_7b_checkpoint` (through the vendored
  `utils/dit_size.py`, header only); `low_vram_warning(is_7b, total_vram_bytes)`.
- `loaded_models.py` — `ModelFileIdentity` (path, size, mtime) and `LoadedModelTracker`: which files the cached models came from.
- `gpu_profiles.py` — `GpuProfile` and `GPU_PROFILES` (8 / 12 / 24 GB: VAE tile size and overlap, BlockSwap for 3B
  and 7B, measured or not), `MANUAL`, `profile_choices`, `find_profile`, `describe_profiles`.
- `pipeline_options.py` — `ColorCorrection`, `AttentionMode`, `PipelineOptions` (defaults LAB, `sdpa`);
  `parse_color_correction` / `parse_attention_mode`; `installed_attention_modes` (the UI's list, by installed package,
  no import); `resolve_attention_mode` (the run's backend from the vendored import flags, `sdpa` with a note otherwise).
- `run_progress.py` — `RunProgress`: the share of a run done (passes by their images, the four `PHASES` by weight,
  batches from the pipeline's callback, DiT blocks inside the DiT phase) and a text, sent to a report function.
- `model_lifecycle.py` — `ModelLifecycle`: drops the vendored cache before a run (Force Reload, other files) and after it
  (Unload, error, interrupt), then releases memory through a function the script passes in.
- `run_inputs.py` (Pillow, numpy) — `select_generated_images` (skip the grid and extras), `select_distinct_images`
  (Batch tab repeats), `to_pipeline_image` (any mode → RGB / RGBA), `group_by_shape`, `split_into_passes`
  (`PASS_BUDGET_PIXELS`, output pixels per input size from the caller), `resolve_seed` / `parse_seed` / `is_unusable_seed`.
- `size_preview.py` — `describe_output_size` (the size line shown before a run, through `UpscaleSettings.target_for`)
  and `txt2img_final_size` (width x height, or the Hires. fix size by Forge's formula and `res_step`).
- `upscale_settings.py` — slider `Bounds` (resolution, scale factor, long edge, noise, tile size and overlap),
  `SizeMode` (Shortest edge / Scale factor / Long edge) and `parse_upscale_settings` → `UpscaleSettings`, clamped to
  those bounds; the script builds its sliders from the same bounds. `UpscaleSettings.target_for(size)` →
  `OutputTarget` (the pipeline's `resolution` / `max_resolution` for one input size, its output size and pixels, and
  `raise_note` for the console and the size line).
- `run_results.py` — `plan_saving` → `SavePlan`; `read_option` (a Forge option with the request's `override_settings`);
  `GenerationParam` keys and `seedvr2_generation_params` (per input: its resolution, the size mode) /
  `failure_generation_params`; `merge_run_images` → `RunImage` list (one per input); `BaseInfotext` (prompt part +
  parameters line, from Forge's infotext or a bare prompt) and `InputDetails` (seed, prompt, base infotext per input),
  built by `describe_generated_inputs` / `describe_uploaded_inputs`; `finish_images` → `ResultImage` list (infotext per
  image, copies of returned inputs keeping `already_saved_as`, infotext embedded in `image.info`) and
  `not_upscaled_comment`.
- `upscale_run.py` (torch, torchvision) — `import_pipeline` (the vendored modules, imported at run time, and
  `attention_availability` from the vendored import flags), `UpscaleRequest` (with `PipelineOptions`), `HostHooks`
  (device, memory release, interrupt flag and progress report from the host), `UpscaleRun.upscale_images` →
  `UpscaleOutcome` (upscaled images by input index, failure text): an `OutputTarget` per input size, passes, the four
  phases with fresh context, memory release after each phase, errors reduced to their type. Interrupt and progress:
  the pipeline's progress callback and `ctx['interrupt_fn']` between batches, and forward pre-hooks on the container
  modules of the VAE and the DiT (`watching_modules`, `container_modules`, `dit_block_types`) inside a batch.
  Under the `cudaMallocAsync` allocator (`keeps_freed_vram`) the same hooks return the unused VRAM pool before each
  decoder stage (`decoder_stages`) while under `TRIM_BELOW_FREE_BYTES` (3 GB) of the GPU is free (`free_vram_bytes`);
  `images_to_tensor` / `tensor_to_images`.

### seedvr2_vendor/core/
- `generation_utils.py` — `setup_generation_context` (devices, dtype), `prepare_runner`, `compute_generation_info`, `load_text_embeddings`.
- `generation_phases.py` — the four phases: `encode_all_batches`, `upscale_all_batches`, `decode_all_batches`, `postprocess_all_batches`.
- `model_configuration.py` — builds the runner from `configs_3b|7b/main.yaml` (chosen by `utils/dit_size.py`, local patch), sets up DiT and VAE, applies tiling, cache reuse.
- `model_loader.py` — loads `.safetensors` / `.gguf` weights.
- `model_cache.py` — process-wide `GlobalModelCache` of DiT/VAE models and runner templates, keyed by id (`forge_dit`, `forge_vae`).
- `infer.py` — `VideoDiffusionInfer`, the diffusion runner. `alpha_upscaling.py` — RGBA support.

### seedvr2_vendor/optimization/
- `memory_manager.py` — moving models/tensors between devices, `release_model_memory`, cleanup.
- `blockswap.py` — swapping DiT blocks to CPU (BlockSwap controls in the Forge UI).
- `gguf_ops.py`, `gguf_dequant.py` — quantized layers. `compatibility.py` — optional-package detection (`gguf`, triton, flash/sage attention), bf16 support.

### seedvr2_vendor/utils/
- `constants.py` — model lookup: the folder passed by the caller first (local patch), then upstream's `./models/SEEDVR2`
  relative to the working directory / ComfyUI `folder_paths`.
- `dit_size.py` — 3B or 7B from the checkpoint header (block count), local addition.
- `color_fix.py` — LAB / wavelet / AdaIN color correction. `debug.py` — logger and timers.
- `downloads.py`, `model_registry.py` — Hugging Face downloads with hashes (not used by the Forge script).

### seedvr2_vendor/models/, seedvr2_vendor/common/, seedvr2_vendor/data/
Model definitions and diffusion math, used as is from upstream.

## Background jobs
None.

## Data
No database. Files:
| What | Where | Who writes | Who reads |
|---|---|---|---|
| Model weights | `<models_path>/SeedVR2/`, else `extensions/<ext>/models/SeedVR2/` (any letter case) | the user | `model_discovery.py` (listing), `dit_size.py` (header only), `model_loader.py` |
| Text embeddings | `pos_emb.pt`, `neg_emb.pt` in the root | committed | `load_text_embeddings` (`torch.load(weights_only=True)`) |
| Upscaled images | `p.outpath_samples` (the tab's output folder) | `seedvr2_script.py` via `images.save_image` | the user |
| In-memory caches | `GlobalModelCache`: DiT/VAE in system RAM, only with "Keep in RAM" (emptied by `seedvr2_neo/model_lifecycle.py` after an "Unload" run, an error, Force Reload or other files); `MODEL_LIFECYCLE`: which files they came from | the pipeline; the script | the pipeline (cache hit); `ModelLifecycle` |

## Kill-switch flags
None: the extension does not act on other people's data. Behaviour is controlled per run from the UI.

## Scripts
`install.py` — the only one; run by Forge, not by hand.

## Dead code
Code that is never called in Forge. Only the owner deletes it.
| What | Why it is dead | Plan |
|---|---|---|
| `__init__.py` (root) | Forge never imports the extension root as a package; it imports `comfy_api` which does not exist in Forge | keep as in upstream except the package rename — upstream diffing (`05-DECISIONS.md`) |
| `seedvr2_vendor/interfaces/` | ComfyUI nodes; import `comfy_api` / `comfy_execution` | keep untouched — upstream diffing (`05-DECISIONS.md`) |
| `seedvr2_vendor/utils/downloads.py`, `model_registry.py` | nothing in Forge calls the download code (`model_registry` is still used for model classes) | keep — needed if auto-download is taken (`06-IDEAS.md`) |
