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
| Runtime | inside Forge Neo, single CUDA GPU (`cuda:0` is hard-coded) |
| Tests | `pytest` outside Forge, no GPU (`requirements-dev.txt`: `pytest`, `gguf`); config in `tests/pytest.ini` |

## Folders
```
<repo root>/                — the extension root (Forge clones it into extensions/)
  scripts/                  — Forge entry point: seedvr2_script.py
  seedvr2_neo/              — our helpers without Forge/Gradio imports (model folder discovery)
  tests/                    — pytest, mirrors seedvr2_neo/ and seedvr2_vendor/; pytest.ini lives here
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
| Test layout | `tests/` mirrors `seedvr2_neo/` and `seedvr2_vendor/`; no `__init__.py` in test folders, default import mode, so test file names must be unique. Planned: `tests/scripts/` with Forge modules (`modules.*`) stubbed in `tests/conftest.py` |
| Test file / stress scenarios file name | `test_<module>.py` / `test_<module>_stress.py` |
| Linter, formatter | none yet; proposed `ruff` on `scripts/` and new code only (not on vendored `seedvr2_vendor/`) |

## Entry points

### Called by Forge Neo
| Hook | Caller | Module | What it does |
|---|---|---|---|
| `install.py` (run as a script) | Forge launcher, every start | `install.py` | `pip install rotary-embedding-torch` if missing |
| `Script.title / show / ui` | Forge, when building the txt2img and img2img tabs | `scripts/seedvr2_script.py` | Adds "SeedVR2 Native Upscaler (24G)" to the Script dropdown; scans the model folder for the DiT/VAE dropdowns, 🔄 rescans it |
| `Script.run(p, *args)` | Forge, when Generate is pressed with this script selected | `scripts/seedvr2_script.py` | txt2img: generate then upscale; img2img: upscale the input image. Returns `Processed` |

### Pages and API for people
None of its own: the UI lives inside the txt2img / img2img tabs.

## Modules

### scripts/
- `seedvr2_script.py` — the whole Forge integration: UI controls and the model refresh button, the model check before a
  run, seed resolution, building the pipeline arguments, cached generation context (`self.runner_cache`), running the four
  phases, saving the result with `-seedvr2` suffix, interruption via `shared.state.interrupted`.

### seedvr2_neo/
- `model_discovery.py` — `scan_models(models_path, extension_root)` → `ModelScan` (absolute folder, DiT and VAE file
  names, defaults); the user-facing note for missing models; the reason a selection cannot run. No Forge imports.

### seedvr2_vendor/core/
- `generation_utils.py` — `setup_generation_context` (devices, dtype), `prepare_runner`, `compute_generation_info`, `load_text_embeddings`.
- `generation_phases.py` — the four phases: `encode_all_batches`, `upscale_all_batches`, `decode_all_batches`, `postprocess_all_batches`.
- `model_configuration.py` — builds the runner from `configs_3b|7b/main.yaml` (chosen by `utils/dit_size.py`, local patch), sets up DiT and VAE, applies tiling, cache reuse.
- `model_loader.py` — loads `.safetensors` / `.gguf` weights.
- `model_cache.py` — process-wide `GlobalModelCache` of DiT/VAE models and runner templates, keyed by id (`forge_dit`, `forge_vae`).
- `infer.py` — `VideoDiffusionInfer`, the diffusion runner. `alpha_upscaling.py` — RGBA support.

### seedvr2_vendor/optimization/
- `memory_manager.py` — moving models/tensors between devices, `release_model_memory`, cleanup.
- `blockswap.py` — swapping DiT blocks to CPU (not exposed in the Forge UI).
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
| In-memory caches | `GlobalModelCache` (models, on GPU), `Script.runner_cache` (generation context) | pipeline / script | pipeline / script |

## Kill-switch flags
None: the extension does not act on other people's data. Behaviour is controlled per run from the UI.

## Scripts
`install.py` — the only one; run by Forge, not by hand.

## Dead code
Code that is never called in Forge. Only the owner deletes it.
| What | Why it is dead | Plan |
|---|---|---|
| `__init__.py` (root) | Forge never imports the extension root as a package; it imports `comfy_api` which does not exist in Forge | keep untouched — upstream diffing (`05-DECISIONS.md`) |
| `seedvr2_vendor/interfaces/` | ComfyUI nodes; import `comfy_api` / `comfy_execution` | keep untouched — upstream diffing (`05-DECISIONS.md`) |
| `seedvr2_vendor/utils/downloads.py`, `model_registry.py` | nothing in Forge calls the download code (`model_registry` is still used for model classes) | keep — needed if auto-download is taken (`06-IDEAS.md`) |
| `finally` block keys `images`, `encoded_latents`, `upscaled_latents` in `seedvr2_script.py` | these keys never exist in `ctx`; only `final_video` is real | fix in B06 |
| `argparse.Namespace` fields in `run()` | most fields are built but the calls pass literals instead | simplify in B07 |
