# SeedVR2 upstream — vendored pipeline reference

Facts about the code in `seedvr2_vendor/` and where it comes from. Each fact has a source.

## Origin
- **Model:** [ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR) (SeedVR / SeedVR2).
- **Pipeline code:** [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler), version 2.5.24,
  commit `4490bd1` (upstream `HEAD`, no `v2.5.24` tag exists; last tag is `v2.5.23`). License: Apache-2.0 (upstream `LICENSE`).
- **Verified live:** `seedvr2_vendor/` (upstream `src/`), `__init__.py`, `configs_3b/main.yaml`, `configs_7b/main.yaml`, `pos_emb.pt`,
  `neg_emb.pt` are identical to upstream `4490bd1` (SHA-256 of files, `diff -r` ignoring line endings) except the "Local patches"
  below. To diff after a sync, compare upstream `src/` with our `seedvr2_vendor/`.
  Ours: `scripts/`, `seedvr2_neo/`, `tests/`, `install.py`, `requirements.txt`, `requirements-dev.txt`, `LICENSE`, `NOTICE`.

## Local patches
Changes we made to vendored files. Re-apply after every upstream sync. Each code change is marked `Local patch (Forge Neo)`.
- `seedvr2_vendor/utils/constants.py`, `find_model_file`: when the caller passes a folder and the file exists there, that path is
  returned first. Why: the upstream scan is relative to the working directory, which in Forge is wherever the launcher
  was started. Test: `tests/seedvr2_vendor/utils/test_constants.py`.
- `seedvr2_vendor/utils/dit_size.py` (new file): reads tensor names from the safetensors / GGUF header and maps the block count
  (32 / 36) to 3B / 7B; unreadable header or another block count → `"7b"` in the name, case-insensitive. Tests: `tests/seedvr2_vendor/utils/test_dit_size*.py`.
- `seedvr2_vendor/core/model_configuration.py`, `_create_new_runner` (+ one import): the config folder comes from
  `choose_dit_size` instead of `"7b" in dit_model`; the chosen size and its source are logged.
- **Package folder renamed** `src/` → `seedvr2_vendor/` (no file content changes from the rename itself: imports inside the
  package are relative). Why: Forge puts every extension folder on `sys.path`, and other SeedVR2 ports also ship a top-level
  `src`; whichever loads first is reused by both. Test: `tests/seedvr2_vendor/test_package_paths.py`.
- `seedvr2_vendor/utils/constants.py`, `get_vae_config_path` (new function) and its use in
  `seedvr2_vendor/core/model_configuration.py`, `_create_new_runner` (+ the import): the VAE config path is built from the
  package's own folder instead of upstream's folder name joined to the repository root. Test: `tests/seedvr2_vendor/test_package_paths.py`.
- `seedvr2_vendor/optimization/memory_manager.py`, `cleanup_dit` and `cleanup_vae`: a model that is not cached is no
  longer moved to the CPU before deletion on any device (upstream skips the move only on MPS). Why: the move only put
  the whole model into system RAM right before dropping it — a ~8 GB RAM peak for 7B fp8 in the default "Unload" mode.
  The GPU memory is freed by dropping the model (`runner.dit = None` / `runner.vae = None`) plus the caller's
  `gc.collect()` and CUDA empty-cache; `release_model_memory` does **not** free it (`param.data.set_()` acts on a
  detached view, the parameter keeps its storage), so any other reference to the model keeps its VRAM held.
  Not unit-tested (tests run without torch); checked live on an 8 GB GPU: "skipping CPU movement" logged, no RAM
  growth at the end of Phase 2, the DiT's VRAM returned before Phase 3.
- `__init__.py` (root, ComfyUI-only): imports `.seedvr2_vendor` instead of `.src`.
- `requirements.txt` is not synced from upstream: ours lists only what the code imports (upstream's adds `peft`, `matplotlib`).

## Common pitfalls
- **Caching requires an offload device.** Upstream's own loaders refuse `cache_model=True` without an offload device
  ("Model caching (cache_model=True) requires offload_device to be set", `seedvr2_vendor/interfaces/dit_model_loader.py`,
  `vae_model_loader.py`); the CLI defaults the offload to `cpu` when caching (`inference_cli.py`, `_parse_offload_device`).
  Our script bypasses the loaders and, without BlockSwap, caches with no offload device: `cleanup_dit` / `cleanup_vae`
  then fall back to the CPU, so cached models wait in system RAM (live: DiT CPU→GPU 11.8 s on the next run), while the
  VAE stays in VRAM during the DiT step. Source: upstream source, `seedvr2_vendor/optimization/memory_manager.py`.
- **BlockSwap and `swap_io_components` require a DiT offload device.** Source: `inference_cli.py` help, `dit_model_loader.py`.
- **Interrupt hook:** `setup_generation_context` sets `ctx['interrupt_fn']` only when ComfyUI is importable; outside ComfyUI it is `None`.
  The phases call `check_interrupt(ctx)` (which calls it) at the top of each batch loop and `progress_callback(current,
  total, …)` after each batch — the same granularity: with batch 1 a whole image's DiT step or tiled decode runs with
  no check. Source: `seedvr2_vendor/core/generation_utils.py` (`check_interrupt`), `generation_phases.py`.
- **Model objects survive weight loading.** `runner.dit` / `runner.vae` exist on the meta device after
  `prepare_runner`; `materialize_model` (in the phase) loads weights with `load_state_dict(state, strict=False,
  assign=True)`, and GGUF replaces only Linear layers (`replace_linear_with_quantized`), so container modules (DiT
  blocks, VAE stages) are the same objects before and after, and hooks set on them before a phase fire inside it. The
  DiT may be wrapped later (`CompatibleDiT`, the model at `.dit_model`); BlockSwap replaces each block's `forward`
  but blocks are still called through `__call__`, so hooks fire. Sources: `core/model_loader.py`
  (`_load_standard_weights`, `_load_gguf_weights`), `core/model_configuration.py`, `optimization/compatibility.py`,
  `optimization/blockswap.py`. The DiT body runs `for block in self.blocks: … module(...)`
  (`models/dit_*/nadit.py`, `gradient_checkpointing` calls the module directly).
- **Tiled VAE decode** calls `self.decoder(...)` once per tile (`tiled_decode` → `slicing_decode` → `_decode`,
  `models/video_vae_v3/modules/attn_video_vae.py`).
- **Attention backends** are imported at module level in `optimization/compatibility.py`, with flags
  `FLASH_ATTN_2_AVAILABLE` (`flash_attn` + `flash_attn_2_cuda`), `FLASH_ATTN_3_AVAILABLE` (`flash_attn_interface`),
  `SAGE_ATTN_2_AVAILABLE` (`sageattention.sageattn_varlen`), `SAGE_ATTN_3_AVAILABLE` (`sageattn3`, or
  `sageattention.sageattn_blackwell`). `validate_attention_mode` falls back to `sdpa` with a warning for an
  unavailable one. Color correction methods of the postprocess phase: `lab`, `wavelet`, `wavelet_adaptive`, `hsv`,
  `adain`, `none` (`core/generation_phases.py`; the ComfyUI node's list in `interfaces/video_upscaler.py`).
- **Config by file name (upstream):** `configs_7b` if `"7b" in dit_model` (case-sensitive), else `configs_3b` — replaced
  here by a local patch. Source: `seedvr2_vendor/core/model_configuration.py`.
- **Model lookup (upstream):** `find_model_file` searches ComfyUI `folder_paths` if present, else `./models/SEEDVR2` relative
  to the working directory, then falls back to the given directory — the given directory now wins (local patch).
  Source: `seedvr2_vendor/utils/constants.py`.
- **Checkpoint layout:** DiT tensor names start with `blocks.<N>.` (`self.blocks` in `seedvr2_vendor/models/dit_*/nadit.py`); 3B has
  32 blocks, 7B has 36 (`num_layers` in `configs_*/main.yaml`). Checked on the official 3B fp16, 3B fp8 and 7B sharp fp8
  safetensors and the 3B Q6_K GGUF (detected as 3B from its header in a live run); 7B GGUF not checked.
- **Import side effects:** importing the pipeline (through `optimization/compatibility.py`) puts stub modules into
  `sys.modules` for `bitsandbytes` (and checks `triton`, `flash_attn`, `xformers`) when they are broken, and probes bf16
  with a small matmul on `cuda:0`. The Forge script imports the pipeline at the start of `run()`, so on the first run
  this happens before the base image is generated. Source: `seedvr2_vendor/optimization/compatibility.py` (shims at
  import, `_probe_bfloat16_support`).
- **The vendored code picks devices itself in places** (`cuda:0` in the bf16 probe, `get_device()` in BlockSwap); on a
  CPU-only Forge BlockSwap fails with an error naming a `--dit_offload_device` CLI flag Forge does not have. Source:
  `seedvr2_vendor/optimization/blockswap.py`, `validate_blockswap_config`.
- **Text embeddings** are loaded with `torch.load(..., weights_only=True)` — no pickled code is executed. Source: `seedvr2_vendor/core/generation_utils.py`.

## Model files
Registry with SHA-256 hashes: `seedvr2_vendor/utils/model_registry.py`; Hugging Face repos `numz/SeedVR2_comfyUI`, `AInVFX/SeedVR2_comfyUI`.

## Related project
[wdhameli0/sd-webui-forge-neo-seedvr2](https://github.com/wdhameli0/sd-webui-forge-neo-seedvr2) (Apache-2.0, v0.3.1) — another Forge Neo
integration of the same upstream (v2.5.23), as an Extras-tab postprocessing script. Read, not run. Worth borrowing (with attribution):
Forge interrupt via `ctx['interrupt_fn']` and progress via `shared.state.job*` (`seedvr2_forge/engine.py`); offload/BlockSwap
validation; `clear_memory(deep=True)` cleanup when not caching; image tiling with feather blending (`tiling.py`); `ensure_rgb`;
2×/3×/4× presets; `install.py` that installs `gguf` and never touches torch; a smoke test. Not to copy: it downloads upstream
source from GitHub at install time without a hash check, and installs unused `peft` / `matplotlib`.

## License
- **Apache-2.0 §4 (redistribution of the Work or Derivative Works, with or without modification)**, text:
  [LICENSE-2.0.txt](https://www.apache.org/licenses/LICENSE-2.0.txt). Conditions: (a) give every recipient a copy of the
  License; (b) modified files carry prominent notices stating that You changed the files; (c) keep, in Source form, all
  copyright, patent, trademark and attribution notices from the Work; (d) if the Work has a NOTICE file, the Derivative Work
  must carry its attribution notices in at least one of: a NOTICE file, the source/docs, or a display where third-party
  notices appear; the NOTICE content is informational and must not modify the License. You may add your own copyright
  statement for your modifications and use other license terms for them, provided the use of the Work complies with the License.
- **Form of the §4(b) notice:** the text only says "prominent notices" in the modified files; it prescribes no format. Whether a
  central list of changes is enough instead of a per-file marker: no primary source found. Safe reading: a short comment in each
  modified file (ours already carry `Local patch (Forge Neo)`), plus the list in this document.
- **Renaming a folder of vendored files as a modification:** no primary source.
- **numz/ComfyUI-SeedVR2_VideoUpscaler** at `4490bd1` ([repo](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)):
  Apache-2.0 (README: "released under the Apache 2.0 license as found in the LICENSE file"). The root listing has no `NOTICE` file.
  Its `LICENSE` is the stock text; the only copyright line is the appendix boilerplate `Copyright 2025 seed`
  ([LICENSE](https://raw.githubusercontent.com/numz/ComfyUI-SeedVR2_VideoUpscaler/4490bd1/LICENSE)), i.e. not a notice naming numz.
- **License headers in the vendored `seedvr2_vendor/` (identical to upstream apart from the local patches):** 64 of the ~80 `.py` files carry
  one. Under `common/`, `data/`, `models/dit_3b|dit_7b`, `models/video_vae_v3` and `core/infer.py` the header is
  `# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates` followed by the Apache-2.0 boilerplate. `video_vae_v3/modules/video_vae.py`
  and `attn_video_vae.py` carry `Copyright (c) 2023 HuggingFace Team` + `Copyright (c) 2025 ByteDance Ltd. and/or its affiliates`
  + "This file has been modified by ByteDance ... on 1st June 2025". These must be kept (§4(c)). The numz-written files
  (`core/` except `infer.py`, `interfaces/`, `optimization/`, `utils/`) have no license header. Source: files in `seedvr2_vendor/`.
- **Third-party origin inside numz files:** `seedvr2_vendor/optimization/gguf_dequant.py` says "Adapted from ComfyUI-GGUF"
  ([city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF), Apache-2.0, `LICENSE` is the stock text with no copyright line).
  `seedvr2_vendor/utils/color_fix.py` cites a paper only (not code).
- **ByteDance-Seed/SeedVR** ([repo](https://github.com/ByteDance-Seed/SeedVR)): Apache-2.0 (README: "SeedVR and SeedVR2 are licensed under
  the Apache 2.0"); the root listing has no `NOTICE`; its `LICENSE` has only the appendix line `Copyright 2025 seed`
  ([LICENSE](https://raw.githubusercontent.com/ByteDance-Seed/SeedVR/main/LICENSE)). The `pos_emb.pt` / `neg_emb.pt` are in both repos.
- **Applying the license to your own work** ([apply-license](https://www.apache.org/legal/apply-license.html), [licensing-howto](https://infra.apache.org/licensing-howto.html)):
  one copy of the license text in a file named `LICENSE` in the top directory (not per file); a `NOTICE` file next to it;
  each original source file "should" carry a short header (appendix boilerplate: `Copyright [yyyy] [name of copyright owner]` +
  the "Licensed under the Apache License, Version 2.0 ..." paragraph, in the file's comment syntax; fill in the brackets). NOTICE is
  "reserved for a certain subset of legally required notifications": keep it minimal, add nothing not legally required, preserve
  copyright notices that were relocated (not removed) from source files, and carry over relevant parts of NOTICE files of bundled
  Apache-2.0 dependencies. These pages are ASF project guidance; for a non-ASF project they are recommendations, not §4 requirements.

## Tiling large images

**Upstream has no tiling of the DiT step.**
- `main` of numz/ComfyUI-SeedVR2_VideoUpscaler still ends at `4490bd1`; the newest release is `v2.5.23`, and no release note mentions
  tiling (GitHub API: [commits](https://api.github.com/repos/numz/ComfyUI-SeedVR2_VideoUpscaler/commits?per_page=15),
  [releases](https://api.github.com/repos/numz/ComfyUI-SeedVR2_VideoUpscaler/releases?per_page=8)). Large-output items only:
  v2.5.18 (CLI `--chunk_size` streaming for long videos, shared memory instead of numpy pickling for a large-output MemoryError)
  and v2.5.23 (VAE slicing division-by-zero guard).
- Open, unmerged [PR #551](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/pull/551) (author xmarre, 39 commits, no maintainer
  review visible) adds optional DiT tiling: `dit_tiled` (off by default), `dit_tile_size` 128 and `dit_tile_overlap` 16 "in latent-space
  pixels", edge-aware blending, single-sample only; the author says it is slower, lower-peak-memory and not guaranteed to match
  full-frame quality (smaller tiles reduce global consistency). Its code was not read.

**Size constraints (vendored code).**
- Input is resized (`NaResize`, mode `side`, no downsampling only) and then padded with zeros to a multiple of 16 on height and width
  (`DivisiblePad((16, 16))`), `seedvr2_vendor/core/generation_utils.py` `prepare_video_transforms`;
  `seedvr2_vendor/data/image/transforms/divisible_crop.py` pads right/bottom with `value=0.0`. Phase 3 trims back to the true size
  (`true_target_dims`, rounded down to even), `core/generation_phases.py` ("Trimming spatial padding").
- 16 = VAE spatial factor 8 (`spatial_downsample_factor`, default 8, `core/infer.py`, `core/model_configuration.py`) x DiT patch
  size `[1, 2, 2]` (`configs_3b/main.yaml`, `configs_7b/main.yaml`): one DiT token covers 16x16 output pixels. Temporal factor 4
  (frames padded to 4n+1; a single image is one frame).

**DiT attention is windowed, and its window size does not grow with the image.**
- Both configs: `window: 32/36 x (4,3,3)`, `window_method` alternating `720pwin_by_size_bysize` / `720pswin_by_size_bysize`
  (regular and half-shifted windows); `rope_type: mmrope3d` (3B), `qk_rope: True` (7B). Code: `models/dit_*/window.py`,
  `models/dit_*/nablocks/attention/mmattn.py`.
- `make_720Pwindows_bysize`: with tokens `h x w`, `scale = sqrt(45*80 / (h*w))`, window = `ceil(round(h*scale) / 3)` by
  `ceil(round(w*scale) / 3)` tokens. So the window area is about 3600/9 = 400 tokens (about 320x320 output pixels; 45x80 tokens =
  720x1280 px at 16 px per token) for any image size, and its aspect ratio follows the image's. The number of windows grows with the
  image (about 3x3 at 720p, 4x4 with the shift, more beyond). Consequence (derived from the code, not measured): attention cost and
  memory grow linearly with the number of tokens, not quadratically.
- RoPE is computed per window: `mmrope3d` is called with `window_shape` (`mmattn.py`), and `NaMMRotaryEmbedding3d.get_freqs` slices
  `vid_freqs[l:l+f, :h, :w]`, i.e. positions start at 0 in each window; there is no absolute image position. Text conditioning is the
  fixed `pos_emb.pt`. Axial frequency tables are capped at 128 tokens per axis (`min(max_height + 4, 128)`, `models/dit_*/rope.py`);
  a window larger than that is not reachable at about 400 tokens per window except at extreme aspect ratios.
- A tile processed alone therefore behaves like the same region of the whole image except: (a) the window grid starts at the tile's
  origin (the half-window shift only exists when the axis has more than one window), (b) the window shape depends on the tile's
  aspect ratio (a square tile gets about 20x20-token windows, a 4:3 image about 17x23), (c) the VAE's tiled encode/decode and any
  per-image color correction see only the tile. No global attention exists, so tiles lose no long-range context the whole-image
  model had beyond one window (and the shifted window); this follows from the code, not from a measurement.

**Related implementations (all pixel-space, each tile through the whole pipeline).**
- [wdhameli0/sd-webui-forge-neo-seedvr2](https://github.com/wdhameli0/sd-webui-forge-neo-seedvr2) (repo Apache-2.0; `seedvr2_forge/tiling.py`
  has no license header): README flow "Lanczos pre-resize, then automatic tile split, then SeedVR2 restoration, then feather tile
  blending", i.e. tiles are cut from the already-enlarged output-size image in pixel space (PIL/NumPy, no latents). Defaults
  `tile_size = 1024`, `tile_padding = 64` (blend size; README suggests 768 / 512 on OOM; zero padding warns of seams). Layout per axis:
  one tile if the axis fits, else `n = ceil(size / tile)` tiles with the surplus overlap spread evenly
  (`overlap = (n*tile - size) // (n-1)`), last tile shifted back to stay full size. Blend: only `blend_padding` pixels at the centre
  of each overlap are feathered with a linear ramp (`np.linspace(1, 0, blend)`, `a*alpha + b*(1-alpha)`), the rest of the overlap is
  hard-cut at the middle; rows merged horizontally first, then vertically. No colour handling in `tiling.py`; in `engine.py` every
  tile gets the ordinary `color_correction` of its own postprocess phase (against that tile's input), tiles of one batch must have
  the same size, `resolution = min(tile.size)`. Sources: `seedvr2_forge/tiling.py`, `seedvr2_forge/engine.py` (read via GitHub).
- [moonwhaler/comfyui-seedvr2-tilingupscaler](https://github.com/moonwhaler/comfyui-seedvr2-tilingupscaler) (MIT-0): input split into
  overlapping tiles, each upscaled by SeedVR2 to a fixed resolution then resized to the target; defaults tile 512, padding 32,
  `mask_blur` 0, blending `auto`; methods multiband (Laplacian pyramid), bilateral, content_aware, linear, simple; a "Chess" tiling
  strategy. README advice against seams: multiband or bilateral, padding 64 or more, Chess. A tile batch is treated by SeedVR2 as
  video frames, so batching unrelated tiles costs some quality. The README says nothing on colour mismatch between tiles.
- [BacoHubo/ComfyUI_SeedVR2_Tiler](https://github.com/BacoHubo/ComfyUI_SeedVR2_Tiler) (splitter + stitcher nodes): `overlap_percent`
  default 10 (range 0-40), `feather_blend` default 1.0 (1 = linear feather, 0 = hard cut); seams: raise overlap to 15-20 %. Values
  from third-party pages ([RunComfy](https://www.runcomfy.com/comfyui-nodes/ComfyUI_SeedVR2_Tiler),
  [comfy.icu](https://comfy.icu/extension/BacoHubo__ComfyUI_SeedVR2_Tiler)), not from its README.
- A user report on [lvladikov/SeedVR2-1.4B discussion 1](https://huggingface.co/lvladikov/SeedVR2-1.4B/discussions/1): seams at 1K-4K
  outputs, attributed by a commenter to the VAE tiled decode (512 / overlap 128). An opinion, not verified behaviour.

## Not verified
- Whether upstream numz has a `NOTICE` file in a place other than the root (the root listing showed none; the GitHub page was
  summarized, a `git ls-tree` check was not done).
- Contents of PR #551 (algorithm, whether "latent-space pixels" means latent cells or 8x pixels); it is unmerged, do not rely on it.
- Peak VRAM of the DiT step versus the image size on an 8 GB GPU (what dominates: per-block activations, window re-ordering copies,
  or the VAE); needs a live measurement at 2K / 3K / 4K with `torch.cuda.max_memory_allocated` after each phase.
- Whether a tile cut in latent space (not pixels) gives the same result as the whole latent: the window-grid dependence above says
  "not identical", the size of the visible effect is unmeasured. Live check: run one 2K image whole, then as DiT-only latent tiles
  with overlap of one window (about 20-27 tokens) and compare the overlap seams and mean colour per tile.
- Colour correction per tile versus whole-image (the `lab` / `wavelet` methods compare against the tile's own input); seam visibility
  on flat regions (sky) is unmeasured.
- wdhameli0 `tiling.py` call site (the caller of `split_image` / `merge_tiles`) was not read; the pipeline-per-tile fact rests on
  `engine.py` `process_tiles` and the README flow.
