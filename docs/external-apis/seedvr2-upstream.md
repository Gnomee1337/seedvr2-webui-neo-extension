# SeedVR2 upstream — vendored pipeline reference

Facts about the code in `src/` and where it comes from. Each fact has a source.

## Origin
- **Model:** [ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR) (SeedVR / SeedVR2).
- **Pipeline code:** [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler), version 2.5.24,
  commit `4490bd1` (upstream `HEAD`, no `v2.5.24` tag exists; last tag is `v2.5.23`). License: Apache-2.0 (upstream `LICENSE`).
- **Verified live:** `src/`, `__init__.py`, `requirements.txt`, `configs_3b/main.yaml`, `configs_7b/main.yaml`, `pos_emb.pt`, `neg_emb.pt`
  are identical to upstream `4490bd1` (SHA-256 of files, `diff -r` ignoring line endings) except the "Local patches" below.
  Ours: `scripts/`, `seedvr2_neo/`, `tests/`, `install.py`, `requirements-dev.txt`.

## Local patches
Changes we made to vendored files. Re-apply after every upstream sync. Each is marked `Local patch (Forge Neo)` in the code.
- `src/utils/constants.py`, `find_model_file`: when the caller passes a folder and the file exists there, that path is
  returned first. Why: the upstream scan is relative to the working directory, which in Forge is wherever the launcher
  was started. Test: `tests/src/utils/test_constants.py`.
- `src/utils/dit_size.py` (new file): reads tensor names from the safetensors / GGUF header and maps the block count
  (32 / 36) to 3B / 7B; unreadable header → `"7b"` in the name, case-insensitive. Tests: `tests/src/utils/test_dit_size*.py`.
- `src/core/model_configuration.py`, `create_new_runner` (+ one import): the config folder comes from
  `choose_dit_size` instead of `"7b" in dit_model`; the chosen size and its source are logged.

## Common pitfalls
- **Caching requires an offload device.** Upstream's own loaders refuse `cache_model=True` without an offload device
  ("Model caching (cache_model=True) requires offload_device to be set", `src/interfaces/dit_model_loader.py`,
  `vae_model_loader.py`); the CLI defaults the offload to `cpu` when caching (`inference_cli.py`, `_parse_offload_device`).
  Our script bypasses the loaders and caches with no offload — models stay in VRAM. Source: upstream source.
- **BlockSwap and `swap_io_components` require a DiT offload device.** Source: `inference_cli.py` help, `dit_model_loader.py`.
- **Interrupt hook:** `setup_generation_context` sets `ctx['interrupt_fn']` only when ComfyUI is importable; outside ComfyUI it is `None`
  and the phases check interruption only through `progress_callback`. Setting `ctx['interrupt_fn']` to a Forge check gives
  interruption inside the phases. Source: `src/core/generation_utils.py`, `check_interrupt`.
- **Config by file name (upstream):** `configs_7b` if `"7b" in dit_model` (case-sensitive), else `configs_3b` — replaced
  here by a local patch. Source: `src/core/model_configuration.py`.
- **Model lookup (upstream):** `find_model_file` searches ComfyUI `folder_paths` if present, else `./models/SEEDVR2` relative
  to the working directory, then falls back to the given directory — the given directory now wins (local patch).
  Source: `src/utils/constants.py`.
- **Checkpoint layout:** DiT tensor names start with `blocks.<N>.` (`self.blocks` in `src/models/dit_*/nadit.py`); 3B has
  32 blocks, 7B has 36 (`num_layers` in `configs_*/main.yaml`). Checked on the official 3B fp16, 3B fp8 and 7B sharp fp8
  safetensors; GGUF files not checked yet.
- **Text embeddings** are loaded with `torch.load(..., weights_only=True)` — no pickled code is executed. Source: `src/core/generation_utils.py`.

## Model files
Registry with SHA-256 hashes: `src/utils/model_registry.py`; Hugging Face repos `numz/SeedVR2_comfyUI`, `AInVFX/SeedVR2_comfyUI`.

## Related project
[wdhameli0/sd-webui-forge-neo-seedvr2](https://github.com/wdhameli0/sd-webui-forge-neo-seedvr2) (Apache-2.0, v0.3.1) — another Forge Neo
integration of the same upstream (v2.5.23), as an Extras-tab postprocessing script. Read, not run. Worth borrowing (with attribution):
Forge interrupt via `ctx['interrupt_fn']` and progress via `shared.state.job*` (`seedvr2_forge/engine.py`); offload/BlockSwap
validation; `clear_memory(deep=True)` cleanup when not caching; image tiling with feather blending (`tiling.py`); `ensure_rgb`;
2×/3×/4× presets; `install.py` that installs `gguf` and never touches torch; a smoke test. Not to copy: it downloads upstream
source from GitHub at install time without a hash check, and installs unused `peft` / `matplotlib`.

## Not verified
- Whether a newer upstream release exists after `4490bd1` (2025-12-24 was the latest commit at the time of checking).
